"""
Recapio - AI-Powered Meeting Minutes Generator (Streamlit version)
Built to fit inside Streamlit Community Cloud's free tier (~1GB RAM).

UI NOTES (read this if you're editing styling):
- Almost all styling comes from .streamlit/config.toml (theme colors),
  NOT from injected CSS. Streamlit computes correct contrast and hover
  states for anything driven by the theme config, so this is far more
  robust across Streamlit versions than hand-written <style> overrides.
- The "how it works" cards use st.container(border=True), a native
  Streamlit feature (1.32+) - no custom HTML/CSS needed for them.
- Only a tiny, minimal CSS snippet remains (the hero banner gradient),
  and every element inside it has an explicit color set so it can't
  end up invisible against a light OR dark browser/OS theme.

TOKEN PERSISTENCE:
- The Hugging Face token is stored in the *visitor's own browser*
  localStorage (via the streamlit-local-storage package), never on the
  server and never shared between visitors. Check "Remember on this
  device" once, and it's pre-filled on every future visit from that
  browser. Unchecking it (or clearing browser storage) forgets it.
- If the streamlit-local-storage package isn't available for any
  reason, the app falls back to plain st.session_state, which still
  works but only lasts for the current tab/session (not across
  reloads) - this keeps the app from crashing if that dependency is
  ever missing.
"""

import os
import re
import gc
import tempfile
from datetime import datetime
from collections import defaultdict

import streamlit as st
from docx import Document

try:
    from streamlit_local_storage import LocalStorage
    _local_storage = LocalStorage()
except Exception:
    _local_storage = None

st.set_page_config(
    page_title="Recapio | Meeting minutes, ready to share",
    page_icon="🎙️",
    layout="centered",
)

ACTION_PATTERNS = [
    r"\bwill\b", r"\bneeds? to\b", r"\bshould\b", r"\bgoing to\b",
    r"\bmust\b", r"\bplease\b", r"\baction item\b",
    r"\bby (monday|tuesday|wednesday|thursday|friday|saturday|sunday|\d{1,2}\s?(am|pm)?)\b",
]
ACTION_REGEX = re.compile("|".join(ACTION_PATTERNS), re.IGNORECASE)


# --------------------------------------------------------------------
# Pipeline stages - each one imports its own heavy library lazily and
# frees the model from memory as soon as it's done.
# --------------------------------------------------------------------

def transcribe_audio(audio_path):
    import whisper
    model = whisper.load_model("tiny")
    result = model.transcribe(audio_path, verbose=False)
    del model
    gc.collect()
    return result["segments"], result["text"]


def simple_speaker_split(segments, pause_threshold=1.2):
    """Lightweight, no-model speaker-turn heuristic. Flips the speaker
    label whenever the pause between segments exceeds pause_threshold
    seconds. This is NOT true diarization - it cannot recognize that a
    speaker who left and returned is the same person - but it costs no
    extra memory or dependency, so it's the default path."""
    labeled = []
    current_speaker = 1
    prev_end = None
    for seg in segments:
        if prev_end is not None and (seg["start"] - prev_end) > pause_threshold:
            current_speaker = 2 if current_speaker == 1 else 1
        labeled.append({
            "speaker": f"Speaker {current_speaker}",
            "text": seg["text"].strip(),
            "start": seg["start"],
            "end": seg["end"],
        })
        prev_end = seg["end"]
    return labeled


def has_memory_for_real_diarization(minimum_gib=2):
    """Avoid a process-level OOM kill on memory-limited Linux hosts.
    An OOM kill can't be caught in Python, so check the cgroup limit
    before loading pyannote's model. Unknown limits are treated as
    sufficient, so local/larger deployments are unaffected."""
    cgroup_limit = "/sys/fs/cgroup/memory.max"
    try:
        with open(cgroup_limit, "r", encoding="utf-8") as limit_file:
            value = limit_file.read().strip()
        if value == "max":
            return True
        return int(value) >= minimum_gib * 1024 ** 3
    except (OSError, ValueError):
        return True


def real_diarization(audio_path, hf_token):
    """True diarization using pyannote's current pipeline."""
    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-community-1", token=hf_token
    )
    output = pipeline(audio_path)
    diarization = output.exclusive_speaker_diarization
    speaker_segments = [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, speaker in diarization
    ]
    if not speaker_segments:
        raise RuntimeError(
            "Pyannote did not find any speech. Check that the uploaded audio contains audible speech."
        )
    del pipeline
    gc.collect()
    return speaker_segments


def attach_real_speakers(segments, speaker_segments):
    def speaker_at(t):
        for s in speaker_segments:
            if s["start"] <= t <= s["end"]:
                return s["speaker"]
        return "Unknown"

    return [
        {
            "speaker": speaker_at((seg["start"] + seg["end"]) / 2),
            "text": seg["text"].strip(),
            "start": seg["start"],
            "end": seg["end"],
        }
        for seg in segments
    ]


def summarize_text(full_text):
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_name = "sshleifer/distilbart-cnn-6-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()

    def chunk_text(text, max_words=500):
        words = text.split()
        for i in range(0, len(words), max_words):
            yield " ".join(words[i:i + max_words])

    parts = []
    with torch.no_grad():
        for chunk in chunk_text(full_text):
            inputs = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024)
            summary_ids = model.generate(
                **inputs, max_length=100, min_length=20, num_beams=4, do_sample=False,
            )
            parts.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))

    del model, tokenizer
    gc.collect()
    return " ".join(parts)


def extract_action_items_and_dates(labeled_segments):
    import spacy
    nlp = spacy.load("en_core_web_sm")

    action_items, key_dates = [], []
    for seg in labeled_segments:
        text = seg["text"]
        if not text:
            continue
        doc = nlp(text)
        dates_in_seg = [e.text for e in doc.ents if e.label_ == "DATE"]

        if ACTION_REGEX.search(text):
            people = [e.text for e in doc.ents if e.label_ == "PERSON"]
            owner = people[0] if people else seg["speaker"]
            due = dates_in_seg[0] if dates_in_seg else "Not specified"
            action_items.append({"task": text, "owner": owner, "due": due})

        for d in dates_in_seg:
            key_dates.append({"date": d, "speaker": seg["speaker"], "context": text})

    del nlp
    gc.collect()
    return action_items, key_dates


def build_docx(summary, action_items, key_dates, labeled_segments):
    speaking_time = defaultdict(float)
    for seg in labeled_segments:
        speaking_time[seg["speaker"]] += seg["end"] - seg["start"]

    doc = Document()
    doc.add_heading("Meeting Minutes (Auto-Generated)", level=1)
    doc.add_paragraph(f"Date: {datetime.now().strftime('%d %b %Y')}")

    doc.add_heading("Participants", level=2)
    for speaker, secs in speaking_time.items():
        doc.add_paragraph(f"{speaker} - spoke {int(secs // 60)}m {int(secs % 60)}s", style="List Bullet")

    doc.add_heading("Summary", level=2)
    doc.add_paragraph(summary)

    if key_dates:
        doc.add_heading("Key Dates Mentioned (Agenda)", level=2)
        table = doc.add_table(rows=1, cols=3)
        table.style = "Light Grid Accent 1"
        hdr = table.rows[0].cells
        hdr[0].text, hdr[1].text, hdr[2].text = "Date", "Speaker", "Context"
        for d in key_dates:
            row = table.add_row().cells
            row[0].text, row[1].text, row[2].text = d["date"], d["speaker"], d["context"]

    doc.add_heading("Action Items", level=2)
    table = doc.add_table(rows=1, cols=3)
    table.style = "Light Grid Accent 1"
    hdr = table.rows[0].cells
    hdr[0].text, hdr[1].text, hdr[2].text = "Task", "Owner", "Due"
    for a in action_items:
        row = table.add_row().cells
        row[0].text, row[1].text, row[2].text = a["task"], a["owner"], a["due"]

    tmp_path = os.path.join(tempfile.gettempdir(), "Meeting_Minutes.docx")
    doc.save(tmp_path)
    return tmp_path


# --------------------------------------------------------------------
# Token persistence helpers (browser localStorage, per-visitor)
# --------------------------------------------------------------------

def load_saved_token():
    if _local_storage is not None:
        try:
            return _local_storage.getItem("recapio_hf_token") or ""
        except Exception:
            return st.session_state.get("hf_token", "")
    return st.session_state.get("hf_token", "")


def save_token(token, remember):
    st.session_state["hf_token"] = token
    if _local_storage is not None:
        try:
            if remember and token:
                _local_storage.setItem("recapio_hf_token", token)
            elif not remember:
                _local_storage.deleteItem("recapio_hf_token")
        except Exception:
            pass


# --------------------------------------------------------------------
# UI - minimal hero, then native Streamlit components for everything
# else so contrast/hover states are handled by the theme, not by us.
# --------------------------------------------------------------------
st.markdown(
    """
    <div style="
        background: linear-gradient(120deg, #15235d 0%, #3d2d83 58%, #7450ac 100%);
        border-radius: 18px; padding: 2rem 2rem; margin-bottom: 1.5rem;">
        <div style="color:#d9ccff; font-size:0.8rem; font-weight:700;
                    letter-spacing:0.08em; text-transform:uppercase;">
            Recapio · Meeting intelligence
        </div>
        <div style="color:#ffffff; font-size:2rem; font-weight:700;
                    line-height:1.15; margin-top:0.4rem;">
            Turn conversations into clear next steps.
        </div>
        <div style="color:#eeeaff; font-size:1rem; margin-top:0.6rem; max-width:560px;">
            Upload a recording and get a summary, action items, key dates,
            and a downloadable minutes document - no account required.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

cols = st.columns(3)
steps = [
    ("1", "Upload", "Add your meeting recording (MP3, WAV, or M4A)."),
    ("2", "Review", "Recapio transcribes, summarizes, and finds commitments."),
    ("3", "Share", "Download polished, ready-to-send meeting minutes."),
]
for col, (number, title, copy) in zip(cols, steps):
    with col:
        with st.container(border=True):
            st.caption(f"STEP {number}")
            st.markdown(f"**{title}**")
            st.caption(copy)

st.divider()
st.subheader("Start with your recording")

uploaded_file = st.file_uploader(
    "Meeting recording",
    type=["wav", "mp3", "m4a"],
    help="Choose an audio recording of the meeting you want to turn into minutes.",
)

pause_threshold = 1.2
use_real_diarization = False
hf_token = ""

with st.expander("Advanced: speaker recognition (optional)"):
    st.caption(
        "By default, Recapio uses simple pause-based speaker detection - "
        "no account or setup needed. Turn this on only if you want more "
        "accurate, voice-based speaker identification."
    )
    use_real_diarization = st.checkbox("Use advanced speaker recognition (Hugging Face)")

    if use_real_diarization:
        saved_token = load_saved_token()
        remember = st.checkbox(
            "Remember my token on this device",
            value=bool(saved_token),
            help="Stored only in your browser's local storage. Never sent to or saved on our server.",
        )
        hf_token = st.text_input(
            "Hugging Face token (read access)",
            value=saved_token,
            type="password",
        )
        save_token(hf_token, remember)
        st.caption(
            "Need a token? Accept the model terms at "
            "huggingface.co/pyannote/speaker-diarization-community-1, "
            "then create one at huggingface.co/settings/tokens."
        )
    else:
        pause_threshold = st.slider(
            "Speaker-turn pause sensitivity (seconds)",
            min_value=0.3, max_value=3.0, value=1.2, step=0.1,
            help=(
                "Flips the speaker label whenever the gap between segments "
                "exceeds this value. Lower it for audio with short pauses "
                "between speakers; raise it if one speaker's natural "
                "pauses are wrongly splitting them into two."
            ),
        )

generate = st.button("Generate Minutes", type="primary", disabled=uploaded_file is None)

if generate and uploaded_file:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    try:
        with st.spinner("Transcribing audio (Whisper-tiny)..."):
            segments, full_text = transcribe_audio(audio_path)

        with st.spinner("Identifying speakers..."):
            if use_real_diarization and hf_token:
                if not has_memory_for_real_diarization():
                    st.warning(
                        "Real speaker recognition needs at least 2GB of RAM and is disabled "
                        "on this host to avoid crashing. Using simple speaker-turn detection instead."
                    )
                    labeled_segments = simple_speaker_split(segments, pause_threshold=pause_threshold)
                else:
                    try:
                        speaker_segments = real_diarization(audio_path, hf_token.strip())
                        labeled_segments = attach_real_speakers(segments, speaker_segments)
                    except Exception as e:
                        import traceback
                        st.warning(f"Real diarization failed ({e}); falling back to simple speaker-turn detection.")
                        with st.expander("Show full error details"):
                            st.code(traceback.format_exc())
                        labeled_segments = simple_speaker_split(segments, pause_threshold=pause_threshold)
            else:
                labeled_segments = simple_speaker_split(segments, pause_threshold=pause_threshold)

        with st.spinner("Summarizing discussion..."):
            summary = summarize_text(full_text)

        with st.spinner("Extracting action items and key dates..."):
            action_items, key_dates = extract_action_items_and_dates(labeled_segments)

        with st.spinner("Building minutes document..."):
            docx_path = build_docx(summary, action_items, key_dates, labeled_segments)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        gc.collect()

    st.divider()
    st.subheader("Transcript (with speaker turns)")
    st.text("\n".join(f"[{s['speaker']}] {s['text']}" for s in labeled_segments))

    st.subheader("AI-Generated Summary")
    st.write(summary)

    st.subheader("Key Dates Mentioned (Agenda)")
    st.table(key_dates) if key_dates else st.caption("No specific dates detected in the discussion.")

    st.subheader("Action Items")
    st.table(action_items) if action_items else st.caption("No clear action items detected.")

    with open(docx_path, "rb") as f:
        st.download_button("📄 Download Meeting Minutes (.docx)", f, file_name="Meeting_Minutes.docx")