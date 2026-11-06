# Membangun RAG KarirFit

RAG dipakai untuk **penjelasan** rekomendasi, bukan untuk peringkat. Untuk setiap karier yang direkomendasikan, sistem mengambil 5 potongan teks dari basis pengetahuan (SKKNI, peta okupasi, profil karier). Potongan itu dikirim ke Gemini sebagai konteks, sehingga setiap klaim dalam penjelasan bisa merujuk ke sumbernya, misalnya `[SKKNI282-J.620100.021.02]`. Ini sesuai Bagian 3.2.4 dan 3.5.4 skripsi.

## 1. Install

```bash
cd backend
pip install -r requirements-rag.txt
```

Model embedding `intfloat/multilingual-e5-base` (sekitar 1 GB) diunduh otomatis saat pertama kali dijalankan. Tidak perlu GPU; laptop biasa cukup untuk sekitar 2.500 potongan teks.

## 2. Kumpulkan dokumen

Letakkan file PDF, DOCX, atau TXT di `backend/kb/docs/`.

| Jenis | Sumber | `doc_type` | `access` |
|---|---|---|---|
| SKKNI bidang TIK (pemrograman, jaringan, keamanan informasi, data science, dll.) | JDIH Kemnaker (jdih.kemnaker.go.id), cari "SKKNI" + bidangnya | `skkni` | `public` |
| Peta Okupasi Nasional bidang TIK | BPSDM Komdigi (dulu Kominfo) | `occupation_map` | `public` |
| RPS / kurikulum PNJ | Dokumen internal prodi | `rps` / `course` | **`private`** |

Dokumen `private` tidak pernah diambil selama `PUBLIC_MODE=1` (FR11). Skripsimu juga hanya mengizinkan dokumen publik dikirim ke Gemini free tier.

PDF hasil scan (berupa gambar) tidak bisa dibaca teksnya. Jalankan OCR dulu, misalnya dengan ocrmypdf.

## 3. Isi `kb/docs/mapping.csv`

Isi satu baris per dokumen, dengan kolom `careers` berisi karier yang relevan, dipisah tanda `|`:

```csv
file,doc_type,careers,source,access,id_prefix
SKKNI_2016_282_Pemrograman.pdf,skkni,software_developer|senior_programmer|mobile_developer,Kepmenaker No. 282/2016 SKKNI Bidang Pemrograman,public,SKKNI282
Peta_Okupasi_TIK.pdf,occupation_map,system_analyst|database_administrator,Peta Okupasi Nasional Bidang TIK,public,POTIK
```

ID karier yang valid: `senior_programmer`, `software_developer`, `system_analyst`, `database_administrator`, `mobile_developer`, `qa_tester`, `ui_ux_designer`, `data_analyst`, `data_engineer`, `data_scientist`, `devops_cloud`, `network_sysadmin`, `cyber_security`, `it_support`, `it_pm`.

## 4. Bangun indeks

```bash
python build_rag.py --no-index   # potong dokumen saja, lalu periksa kb/chunks.jsonl
python build_rag.py              # potong + buat embedding + simpan kb/index.faiss dan kb/chunks.json
```

Cara dokumen dipotong:
- **SKKNI:** satu potongan per unit kompetensi (dimulai dari `KODE UNIT`). Hanya bagian deskripsi, elemen, dan kriteria unjuk kerja yang bisa diambil. Bagian *Batasan Variabel* dan *Panduan Penilaian* tetap disimpan, tetapi ditandai tidak untuk diambil karena isinya hampir sama di semua unit.
- **Jenis lain:** dipotong per paragraf, maksimal 1.500 karakter dengan overlap 200 karakter.
- **Profil karier** dari `model.json` ditambahkan otomatis.
- Nomor halaman serta header dan footer yang berulang dihapus.

## 5. Tentukan ambang τ (30 kueri pengembangan)

```bash
python tune_tau.py sheet   # membuat kb/tau_labels.csv: 2 kueri per karier, peringkat 1–5, 8, 15, 30
# buka kb/tau_labels.csv, isi kolom `relevant` dengan 1/0 sesuai kriteria tertulismu
python tune_tau.py fit     # menghitung τ terbaik (balanced accuracy)
```

Masukkan hasilnya ke `backend/.env`, misalnya `RAG_THRESHOLD=0.82`. Kalau mau memakai kueri sendiri, ubah `kb/dev_queries.csv`, lalu jalankan `sheet` lagi. Kueri ini hanya untuk pengembangan; jangan dipakai lagi di evaluasi akhir.

## 6. Jalankan

```bash
# backend/.env
GEMINI_API_KEY=...        # wajib supaya penjelasan dibuat oleh LLM
RAG_THRESHOLD=0.82
PUBLIC_MODE=1

uvicorn main:app --reload --port 8000
```

Cek di http://localhost:8000/api/health. Hasilnya harus `"rag": true`. Di website, bagian penjelasan akan menampilkan daftar **Sumber** di bawah teks.

Kalau tidak ada potongan yang lolos τ untuk suatu karier, penjelasan tetap dibuat dari data sistem. Website lalu menampilkan pemberitahuan bahwa bukti di basis pengetahuan tidak cukup (FR8).
