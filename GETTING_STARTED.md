# Getting Started: collecting crowd clips from a search prompt

This guide is for a collaborator setting up on their own machine. You type a search prompt such as
`"Kumbh Mela crowd aerial drone view"`. The pipeline finds matching YouTube videos and downloads them.
Gemini then scores every frame and keeps only **dense crowds filmed from height**. Each kept segment
becomes a clip in a `dataset/` folder.

You don't need Claude Code. You need about 30 minutes to set up and a few GB of free disk space.
The commands below are for Windows PowerShell. Mac/Linux differences are noted where they matter.

---

## 1. Install the tools (once)

| Tool | Windows | Mac | Check it works |
|---|---|---|---|
| Python 3.10+ | `winget install Python.Python.3.12`, or the python.org installer with **"Add python.exe to PATH"** ticked | `brew install python@3.12` | `python --version` |
| Git | `winget install Git.Git` | `brew install git` | `git --version` |
| FFmpeg | `winget install ffmpeg` | `brew install ffmpeg` | `ffmpeg -version` |

**Close and reopen the terminal after installing**, or the new commands won't be found.

## 2. Get the code and install dependencies (once)

```powershell
git clone https://github.com/Samarth2405/auto-data-collect.git
cd auto-data-collect

python -m venv .venv
.venv\Scripts\Activate.ps1          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt     # ~5 min: pulls PyTorch (~500 MB)
```

If PowerShell says *"running scripts is disabled"*, run this once and then activate again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

**Each time you open a new terminal**, `cd` into the folder and run `.venv\Scripts\Activate.ps1` again.
`(.venv)` at the start of the prompt means the environment is active.

## 3. Get your own API keys (once, free)

Use your own keys, not someone else's. Quotas are per key, so two people sharing one key run out twice as fast.

1. **YouTube Data API v3:** go to [console.cloud.google.com](https://console.cloud.google.com). Create a project, then
   open **APIs & Services → Library**, search "YouTube Data API v3" and click **Enable**. Then go to
   **Credentials → Create credentials → API key**.
2. **Gemini:** go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey) and click **Create API key**.

In the `auto-data-collect` folder, run `notepad .env`. Click **Yes** to create the file, paste these two lines
with your keys, and save:

```
YOUTUBE_API_KEY=AIza...your-youtube-key...
GEMINI_API_KEY=...your-gemini-key...
```

The file must be named exactly `.env`, not `.env.txt`. Never send this file to anyone or commit it.

## 4. Check the install

```powershell
python test_smoke.py
```

The last line should be `All smoke tests passed.` This check needs no internet and no keys.

---

## 5. Extract data from a prompt

### 5a. Preview first (free, no downloads)

```powershell
python stage1_search.py "Lalbaug Ganpati crowd aerial drone view" --max-results 5 --max-duration 300 --drop-ground --out crowd_output_india/mumbai-lalbaug-ganpati
```

This prints the videos it found and saves them to
`crowd_output_india/mumbai-lalbaug-ganpati/discovery.json`. It uses no Gemini calls. The mark before each
video ID is `^` when the title or description suggests it was filmed from height, and `?` when it's unclear.
If the titles look like vlogs, walking tours, or empty streets, change the prompt and preview again.

### 5b. Run it for real

Run the same command again and add the four flags at the end:

```powershell
python stage1_search.py "Lalbaug Ganpati crowd aerial drone view" --max-results 5 --max-duration 300 --drop-ground --out crowd_output_india/mumbai-lalbaug-ganpati --min-persons 0 --full-pipeline --dataset dataset --location mumbai-lalbaug-ganpati
```

What each part does:

| Part | Meaning |
|---|---|
| `"..."` | **The prompt.** The YouTube search text. |
| `--max-results 5` | How many videos to try. Start with 3–5. |
| `--max-duration 300` | Skip videos longer than 5 minutes. Every second of video costs one Gemini call. |
| `--drop-ground` | Skip videos whose title or description says they're shot at ground level (walking tour, vlog, POV). |
| `--out crowd_output_india/<slug>` | Working folder with the downloads and caches. Use a **separate folder per location**. |
| `--min-persons 0` | **Always include this.** The local person detector can't see people from above, so without this flag it throws away the frames you want. |
| `--full-pipeline` | Download, score with Gemini, cut segments, and export the clips. |
| `--dataset dataset` | Where the final clips go. Use the **same** `dataset` folder for every run. |
| `--location <slug>` | Label for this place, e.g. `mumbai-lalbaug-ganpati`. Use the same value as in `--out`. |

### Writing good prompts

- Use **place or event + crowd word + height word**, for example:
  - `"Chhath Puja ghat crowd drone view"`
  - `"Puri Rath Yatra aerial drone"`
  - `"Lalbaug cha raja darshan queue from above"`
- Height words that work: `aerial`, `drone view`, `from above`, `top view`, `overhead`, `rooftop`, `from FOB`.
- Hindi words often find local uploads: `bheed`, `भीड़`, `mela`, `yatra`, `visarjan`.
- Avoid `walking tour`, `vlog`, and `POV`. These are almost always shot at ground level and get filtered out.

### Choosing the location slug

The format is `city-place`: lowercase, words separated by hyphens. The first word is the city.
Clip IDs are built from the first distinctive word after the city. For example, `mumbai-lalbaug-ganpati`
gives clips `IND_LAL_0001`, `IND_LAL_0002`, and so on. Common words like market, station, temple,
ghat and chowk are skipped.

**Rules to keep the datasets mergeable:**

1. **Agree on locations up front.** Two people must never use the same slug or the same 3-letter code.
   These are already taken:
   `amritsar-golden-temple` (GOL), `delhi-chandni-chowk` (CHA), `india-election-rally` (ELE),
   `india-protest` (PRO), `india-roadshow` (ROA), `jaipur-johari-bazaar` (JOH),
   `kolkata-howrah-bridge` (HOW), `mathura-holi` (HOL), `mumbai-dadar-market` (DAD),
   `mumbai-railway-station` (RAI), `patna-chhath-puja` (CHH), `prayagraj-kumbh-mela` (KUM),
   `varanasi-dashashwamedh-ghat` (DAS).
2. **Run one prompt per location slug.** Numbering restarts at `0001` on every run, so a second, different
   prompt with the same `--location` overwrites the first run's clips. If you want more videos for a place,
   raise `--max-results` in that one run. Rerunning the *exact same* command is safe, because it picks up from
   the cache.

### How long it takes (free tier)

With `--min-persons 0`, every second of video costs one Gemini call, and the free tier allows 15 calls a minute.
So **1 minute of video takes about 4 minutes**. Five 3-minute videos take about an hour. You can leave it running.

The free tier also has a daily cap. If you hit it, or any other quota limit, the run stops with a
`[stage4] stopping batch` message. **Run the exact same command later.** Frames that are already scored are cached,
so nothing is paid for twice. A paid Gemini key can go much faster: add `--rpm 300 --workers 8`.

YouTube search costs about 100 units per prompt out of 10,000 a day, so it's rarely the limit.

### Running many prompts overnight

Create a file `my_runs.bat` in the project folder with one line per location, then double-click it:

```bat
@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat

python stage1_search.py "Lalbaug Ganpati crowd aerial drone view" --max-results 5 --max-duration 300 --drop-ground --min-persons 0 --full-pipeline --out crowd_output_india\mumbai-lalbaug-ganpati --dataset dataset --location mumbai-lalbaug-ganpati
python stage1_search.py "Puri Rath Yatra crowd aerial drone" --max-results 5 --max-duration 300 --drop-ground --min-persons 0 --full-pipeline --out crowd_output_india\puri-rath-yatra --dataset dataset --location puri-rath-yatra

echo Done.
pause
```

If one video fails, the run carries on with the others.

---

## 6. What you get

```
dataset/
  manifest.csv                                   one row per clip; open it in Excel
  mumbai-lalbaug-ganpati/
    elevated/IND_LAL_0001/clip.mp4               the crowd segment
                          poster.jpg             its densest frame
                          metadata.json          scores, timestamps, source video, channel
    aerial/IND_LAL_0002/...
crowd_output_india/mumbai-lalbaug-ganpati/       working files (full videos, frames, caches): large
```

**Zero clips for a location** means no frame was both dense (Gemini score 7 or higher) and filmed from
height. To see why, open `crowd_output_india/<slug>/results.json`. Then try a different prompt with a **new**
slug.

Every clip has `"rights_cleared": false`. These are YouTube uploads, so don't publish or redistribute them.

## 7. Sending your results back

Zip the **`dataset`** folder and share it, for example on Google Drive. Don't send:

- `.env` (your keys)
- `crowd_output_india/` (full downloads, many GB; keep it locally so reruns stay free)

---

## Troubleshooting

| Message | Fix |
|---|---|
| `'python' is not recognized` | Reinstall Python with "Add to PATH" ticked, or use `py` instead of `python`. |
| `ffmpeg not found` | Install FFmpeg (step 1), then **reopen the terminal**. |
| `YOUTUBE_API_KEY environment variable not set` | `.env` is missing, in the wrong folder, or saved as `.env.txt`. |
| HTTP 403 `accessNotConfigured` | YouTube Data API v3 isn't enabled on your Cloud project (step 3). |
| HTTP 403 `quotaExceeded` (YouTube) | The daily 10,000 units are used up. Wait for the reset at midnight Pacific time. |
| Videos fail to download | `pip install -U yt-dlp` (YouTube changes often). |
| `[stage4] stopping batch` | Gemini quota or key problem. Check the key, wait, then rerun the same command. |
| `ModuleNotFoundError` | The virtual environment isn't active. Run `.venv\Scripts\Activate.ps1`. |

For more options (tuning thresholds, exporting ground-level crowds too, how the stages work), see
[README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md).
