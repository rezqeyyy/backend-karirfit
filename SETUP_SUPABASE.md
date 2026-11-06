# Setup Supabase, Groq, Render, dan Vercel

Supabase dipakai untuk tiga hal:
- **Login:** Supabase Auth dengan email dan kata sandi.
- **Riwayat tes:** tabel `assessments`, yang hanya bisa dibaca pemilik akunnya.
- **Log evaluasi backend:** tabel `log`, yang diisi FastAPI lewat `DATABASE_URL`.

## 1. Buat proyek Supabase
1. Buka supabase.com → **New project**. Pilih region **Singapore** dan simpan *database password*-nya.
2. Buka **SQL Editor** → **New query**, tempel isi `supabase/schema.sql` (ada di zip frontend), lalu **Run**.

## 2. Atur login
1. Buka **Authentication → Providers → Email** dan pastikan statusnya aktif.
   - Untuk uji coba dan evaluasi dengan peserta, matikan **Confirm email** supaya peserta bisa langsung masuk setelah mendaftar.
   - Kalau tetap aktif, peserta harus mengklik tautan di email dulu.
2. Buka **Authentication → URL Configuration**:
   - **Site URL:** URL Vercel-mu, misalnya `https://karirfit-ecru.vercel.app`
   - **Redirect URLs:** tambahkan `https://karirfit-ecru.vercel.app/masuk` dan `http://localhost:3000/masuk`

## 3. Ambil kunci
- **Project Settings → API**: catat **Project URL** dan **anon public key**.
- **Project Settings → Database → Connection string → URI**, pilih **Transaction pooler**. Ganti `[YOUR-PASSWORD]` dengan *database password* dari langkah 1.

## 4. Vercel (frontend)
Buka **Settings → Environment Variables**. Untuk ketiga variabel, pilih **Config** dan aktifkan untuk semua environment:
```
NEXT_PUBLIC_API_URL           = https://<backend>.onrender.com
NEXT_PUBLIC_SUPABASE_URL      = <Project URL>
NEXT_PUBLIC_SUPABASE_ANON_KEY = <anon public key>
```
Lalu **Redeploy**. *Anon key* memang aman dipakai di browser; keamanan data dijaga oleh *row-level security* di `schema.sql`.

## 5. Render (backend)
Buka **Environment** dan isi:
```
SUPABASE_URL       = <Project URL>
SUPABASE_ANON_KEY  = <anon public key>
REQUIRE_AUTH       = 1
DATABASE_URL       = <URI transaction pooler dari langkah 3>
GROQ_API_KEY       = <dari console.groq.com → API Keys>
GROQ_MODEL         = openai/gpt-oss-120b
GEMINI_API_KEY     = <opsional, untuk penjelasan>
ALLOWED_ORIGINS    = https://karirfit-ecru.vercel.app
ADMIN_TOKEN        = <bebas, rahasia>
EVAL_MODE          = 0   (ubah ke 1 selama evaluasi dengan peserta)
```
Kalau sebelumnya kamu membuat database Postgres di Render, sekarang tidak diperlukan lagi.

## 6. Cek
- `https://<backend>.onrender.com/api/health` harus menampilkan `"auth_required": true`, `"chat": true`, dan `"storage": "postgres"`.
- Buka website, klik **Mulai tes** → diarahkan ke **Masuk** → **Daftar** → isi tes → hasil.
  - Di Supabase, **Table Editor → assessments** harus berisi satu baris baru.
  - Klik **Tanya asisten karier** untuk mencoba chatbot.

## Catatan
- Kalau `NEXT_PUBLIC_SUPABASE_*` belum diisi, halaman tes menampilkan pesan bahwa login belum dikonfigurasi.
- Riwayat disimpan otomatis setiap kali pengguna menyelesaikan tes (tombol **Lihat hasil** di langkah terakhir). Membuka hasil lama dari halaman **Riwayat** tidak membuat baris baru.
- Proyek Supabase gratis bisa dijeda kalau lama tidak dipakai. Sebelum sesi evaluasi, buka dashboard dan pastikan proyeknya aktif.
