# EduSpark — صوت «شرح المعلم»: العربي بصوت المعلم (Chatterbox مستنسخ من Kokoro am_michael) + الإنجليزي بـ Kokoro (am_michael / am_puck)
import json, re, sys, os, numpy as np, soundfile as sf, torch, torchaudio.functional as AF
from kokoro import KPipeline
units = [u.strip() for u in sys.argv[1].split(',') if u.strip()]; shard, nsh = int(sys.argv[2]), int(sys.argv[3]); out = 'out'; os.makedirs(out, exist_ok=True)
allj = []
for un in units:
    for j in json.load(open(f'voice_jobs/{un}.json', encoding='utf-8'))['jobs']: allj.append(dict(j, id=f"{un}__{j['id']}"))
allj.sort(key=lambda j: j.get('en', False))  # العربي (الأثقل) يتوزع أولاً بالتساوي
jobs = [j for i, j in enumerate(allj) if i % nsh == shard]
kp = KPipeline(lang_code='a')
def en(t, voice='am_michael', speed=0.88):
    return np.concatenate([np.asarray(x) for _, _, x in kp(t, voice=voice, speed=speed)])
def level(a, db=-20.0):
    r = np.sqrt(np.mean(a ** 2)) + 1e-9
    return (a * (10 ** (db / 20) / r)).astype(np.float32)
need_ar = any(not j.get('en') for j in jobs)
if need_ar:
    REF = 'teacher_ref.wav'
    sf.write(REF, en("Hello everyone, welcome back to our class. Today we are going to learn something new and exciting together. Listen carefully and try your best.", speed=1.0), 24000)
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    m = ChatterboxMultilingualTTS.from_pretrained(device='cpu')
def light(t):  # تشكيل خفيف: نحذف حركات أواخر الكلمات (وقف طبيعي) ونبقي الشدة وحركات داخل الكلمة
    return re.sub(r'[ً-ِْ]+(?=[\s.,،؟?!:؛)]|$)', '', t)
def ar(t):
    for seed, tt in ((7, light(t)), (11, light(t)), (7, t), (3, light(t) + ' .')):
        try:
            torch.manual_seed(seed)
            w = m.generate(tt, language_id='ar', audio_prompt_path=REF, exaggeration=0.6, cfg_weight=0.45)
            return AF.resample(w, m.sr, 24000).squeeze(0).numpy()
        except Exception as e: print('retry', seed, repr(e)[:80], flush=True)
    return np.zeros(2400, np.float32)
LAT = re.compile(r"([A-Za-z][A-Za-z0-9'’.,!?/\- …]*[A-Za-z0-9.!?'’]|[A-Za-z])")
def chunks(t, n=170):
    res, cur = [], ''
    for s in re.split(r'(?<=[.؟!،:])\s+', t):
        if len(cur) + len(s) > n and cur: res.append(cur); cur = s
        else: cur = (cur + ' ' + s).strip()
    res += [cur] if cur else []
    out2 = []
    for c in res:  # لا مقاطع قصيرة جداً (تسبب خطأ في Chatterbox)
        if out2 and len(c) < 30: out2[-1] += ' ' + c
        else: out2.append(c)
    if len(out2) > 1 and len(out2[0]) < 30: out2[1] = out2[0] + ' ' + out2[1]; out2.pop(0)
    return out2
sil = lambda s: np.zeros(int(24000 * s), np.float32)
for j in jobs:
  try:
      if j.get('en'):
          a = level(en(j['text'], j.get('voice', 'am_michael'), 0.85 if len(j['text'].split()) < 4 else 0.9))
          parts = [sil(.08), a, sil(.12)]
      else:
          parts = []
          for i, seg in enumerate(LAT.split(j['text'])):
              seg = seg.strip(' ،,:؛')
              if not seg or not re.search(r'\w', seg): continue
              if i % 2: parts += [sil(.3), level(en(seg, speed=0.8 if len(seg.split()) < 3 else 0.88), -18.5), sil(.3)]
              else:
                  for c in chunks(seg): parts += [level(ar(c)), sil(.35)]
      f = f"{out}/{j['id']}.wav"; sf.write(f, np.concatenate(parts), 24000)
      os.system(f'ffmpeg -loglevel error -y -i "{f}" -ac 1 -af "acompressor=threshold=-20dB:ratio=3:attack=5:release=80,loudnorm=I=-14:TP=-1" -b:a {"48k" if j.get("en") else "64k"} "{f[:-4]}.mp3" && rm "{f}"')
      print('done', j['id'], flush=True)
  except Exception as e: print('FAILED', j['id'], repr(e)[:120], flush=True)
