# Deploy backend FastAPI ke Render

## Persiapan
1. Push seluruh folder `karirfit/` ke repositori GitHub, termasuk `render.yaml` di akar repo dan `backend/model.json` hasil Colab.
2. Pastikan `backend/model.json` sudah model terbaru, karena Render membaca file dari repo.

## Cara 1: Blueprint (otomatis, disarankan)
1. Buka Render Dashboard, pilih **New → Blueprint**, lalu pilih repo kamu.
2. Render membaca `render.yaml` dan membuat dua layanan:
   - `karirfit-api`: web service FastAPI (Free, region Singapore)
   - `karirfit-db`: Postgres (Free) untuk menyimpan log evaluasi
3. Isi variabel yang diminta:
   - `ALLOWED_ORIGINS`: URL frontend Next.js, misalnya `https://karirfit.vercel.app`. Untuk tes lokal, tambahkan juga `http://localhost:3000` (pisahkan dengan koma).
   - `GEMINI_API_KEY`: wajib kalau penjelasan ingin dibuat oleh LLM.
   - `GROQ_API_KEY`: opsional.
4. Klik **Apply**. Setelah build selesai, buka `https://karirfit-api.onrender.com/api/health`. Hasilnya harus:
   `"ok": true`, `"storage": "postgres"`, dan `demand_source` = `demand_measured.csv`.

## Cara 2: Manual
Pilih **New → Web Service**, lalu isi:
- Build command: `pip install -r backend/requirements.txt`
- Start command: `cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/api/health`
- Environment: variabel yang sama seperti di Cara 1, ditambah `DATABASE_URL` dari database Postgres yang kamu buat.

## Sambungkan frontend
Di Vercel atau di `frontend/.env.local`:
```
NEXT_PUBLIC_API_URL=https://karirfit-api.onrender.com
```

## Batasan paket Free (penting untuk evaluasi skripsi)
| Batasan | Dampak | Yang sudah disiapkan |
|---|---|---|
| Server tidur setelah 15 menit tanpa trafik dan butuh sekitar 1 menit untuk bangun | Pengunjung pertama menunggu | Frontend menampilkan "Server sedang dinyalakan…" dan mencoba ulang otomatis sampai sekitar 2 menit |
| Filesystem hilang setiap restart, redeploy, atau saat server tidur | Log SQLite akan hilang | Backend memakai Postgres lewat `DATABASE_URL` |
| Postgres Free kedaluwarsa 30 hari setelah dibuat | Log rating dan persetujuan peserta bisa hilang | Ekspor log sebelum 30 hari: `curl -H "X-Admin-Token: <token>" https://karirfit-api.onrender.com/api/export > log.json`. Token ada di tab Environment |
| RAM 512 MB | Model embedding RAG (e5-base + PyTorch) tidak muat | Tanpa RAG, backend tetap jalan dan penjelasan dibuat tanpa konteks dokumen |

**Saran waktu evaluasi:** selama pengambilan data dengan 23 peserta, naikkan instance ke paket berbayar yang tidak tidur, dan jalankan RAG di instance dengan RAM 2 GB (Standard). Setelah evaluasi selesai, kembalikan ke Free.
