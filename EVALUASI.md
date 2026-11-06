# Panduan evaluasi KarirFit (Bab III Bagian 3.6)

| Uji | Bagian skripsi | Alat |
|---|---|---|
| Akurasi peringkat (Hit@1, Hit@3, MRR vs cosine) | 3.6.2 | Notebook bagian 10 (profil uji < 3 skill diganti otomatis) |
| Sensitivitas α dan ablasi | 3.6.3 | Notebook bagian 11 |
| Triangulasi kebutuhan skill (SKKNI/okupasi dan O*NET) | 3.2.2, 3.2.5 | Notebook bagian 7b |
| Rating relevansi oleh peserta | 3.6.4 | Website (EVAL_MODE=1) |
| Peringkat normal vs tanpa minat (α = 1), acak tanpa label | 3.6.4 | Website, panel evaluasi |
| Dukungan bukti RAG vs non-RAG (60 profil) | 3.6.5 | `backend/eval_rq5.py` |
| Dukungan bukti dari penjelasan peserta | 3.6.5 | Otomatis: setiap penjelasan dicek `claims.py`, kembaran non-RAG dicatat |
| Kelengkapan 4 bagian dan ketepatan gap | 3.6.1 | `claims.py` (kolom `four_parts`, `gaps_correct`) |
| SUS | 3.6.6 | Website, panel evaluasi |

## 1. Selama pengambilan data dengan peserta
1. Di Render → Environment: `EVAL_MODE=1`, `GEMINI_API_KEY`, dan `ADMIN_TOKEN`.
2. Peserta mengisi tes dan **mencentang persetujuan** di langkah 5.
3. Di halaman hasil muncul panel **"Evaluasi untuk peserta penelitian"** berisi:
   - penanda kemajuan rating relevansi untuk 7 karier yang direkomendasikan,
   - perbandingan Peringkat A/B (normal vs α = 1, urutan acak, label disembunyikan di token bertanda tangan),
   - kuesioner SUS dengan 10 butir.
4. Setiap penjelasan yang dilihat peserta dicek otomatis. Dengan `EVAL_MODE=1`, prompt yang sama juga dikirim **tanpa potongan dokumen** di latar belakang, lalu dicek dan dicatat, tapi tidak ditampilkan (3.5.6).
5. Setelah selesai, buka `/admin` di website, masuk dengan `ADMIN_TOKEN`, lalu **Unduh log evaluasi (JSON)**.

Isi log menurut `kind`:
- `rating`: skor 1–5 per karier
- `compare_choice`: `prefers_normal` (true/false) untuk proporsi peserta yang memilih peringkat normal
- `sus`: `score` 0–100, dengan target rata-rata ≥ 68
- `explain`: kondisi RAG, dengan `check.evidence_support`, `four_parts`, `gaps_correct`, teks, potongan dokumen, dan model
- `explain_norag`: kondisi non-RAG dengan isi yang sama

## 2. Evaluasi RQ5 dengan 60 profil uji
```bash
cd backend
pip install -r requirements-rag.txt scipy
python build_rag.py                      # indeks harus siap
python eval_rq5.py heldout_ads.csv       # heldout_ads.csv dari karirfit_output.zip (notebook)
```
Hasilnya ada di `backend/eval_rq5/`:
- `summary.json`: rata-rata dukungan bukti RAG dan non-RAG, p-value Wilcoxon, kelengkapan, ketepatan gap, dan status ketiga kriteria penerimaan
- `manual_coding.csv`: 30 pasangan buta untuk koding manual. **Jangan buka `manual_key.csv` sebelum koding selesai**

## 3. Pemeriksa klaim (`claims.py`)
Klaim yang dicek otomatis:
- sitasi `[ID]` benar-benar diambil
- kode unit SKKNI ada di potongan dokumen
- skor `NN/100`
- persentase kecocokan kompetensi
- persentase permintaan per skill (toleransi ±1 poin)
- status skill: di bagian 2 harus dimiliki, di bagian 3 harus gap
- mata kuliah yang disebut sebagai bukti harus pernah diambil pengguna

Klaim lain (misalnya kalimat umum) tidak dihitung dan menjadi bahan koding manual.
