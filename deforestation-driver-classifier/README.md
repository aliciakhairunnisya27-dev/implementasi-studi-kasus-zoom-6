# Atribusi Penyebab Langsung Peringatan Deforestasi dengan Citra Satelit dan AI

Studi kasus + implementasi referensi: klasifikasi **penyebab langsung** (*direct driver*)
dari setiap peringatan deforestasi (*deforestation alert*) menggunakan citra Sentinel-1
(radar) dan Sentinel-2 (optik) yang diproses dengan *convolutional neural network* (CNN).

> **Catatan penting.** Repo ini adalah implementasi **edukatif** yang terinspirasi oleh
> pendekatan umum atribusi penyebab deforestasi berbasis AI. Repo ini **bukan** replikasi
> model atau data penelitian Dr. Bart Schlachter. Data bawaan adalah **sintetis**, sehingga
> angka akurasi hanya membuktikan bahwa pipeline berjalan, **bukan** performa di lapangan.

## 1. Studi kasus

**Skenario.** Sebuah unit pengelola kawasan hutan di lanskap tropis (misalnya Sumatra atau
Kalimantan) menerima ratusan peringatan deforestasi setiap bulan dari sistem peringatan
berbasis satelit. Peringatan hanya mengatakan *"di sini tutupan pohon hilang"*, bukan
*"mengapa"*. Tim patroli terbatas, sehingga harus memutuskan lokasi mana yang diperiksa
lebih dulu.

**Pertanyaan.** Dapatkah setiap peringatan diberi label penyebab langsung secara otomatis
sehingga tim bisa memprioritaskan, misalnya pertambangan atau jalan baru di dalam kawasan
lindung, dibanding pembukaan kebun kecil yang sudah lama dikenal?

**Kelas penyebab (7 kelas):**

| Kelas | Ciri khas pada citra |
|---|---|
| `pertanian_skala_kecil` | Beberapa bidang kecil, vegetasi campuran, bersebelahan |
| `pertanian_skala_besar` | Bidang besar, berbentuk geometris, tanah terbuka seragam |
| `jalan` | Area sempit dan memanjang |
| `penebangan_selektif` | Titik-titik kecil tersebar, sering dekat jalan sarad |
| `pertambangan` | Area besar tidak beraturan, tanah terbuka sangat terang di SWIR |
| `kebakaran` | Bekas bakar gelap (NIR turun, SWIR naik), tepi tidak beraturan |
| `gangguan_alami` | Mis. banjir/longsor: memanjang di sepanjang sungai, radar sangat gelap |

**Alur kerja.**

```
Peringatan deforestasi (lokasi + mask area berubah)
        |
        v
Ambil patch 64x64 piksel di sekitar peringatan
   - Sentinel-2: B2, B3, B4, B8, B11, B12 (optik)
   - Sentinel-1: VV, VH (radar, tembus awan)
   - Mask peringatan (1 kanal)        => 9 kanal total
        |
        v
CNN (3 blok konvolusi + global pooling) --> probabilitas 7 kelas
        |
        v
Prioritas patroli + statistik penyebab per wilayah
```

**Mengapa dua sensor?** Citra optik kaya informasi vegetasi tetapi terhalang awan. Radar
Sentinel-1 tetap mengamati saat berawan. Generator data sintetis sengaja menutupi sebagian
data optik dengan awan agar model belajar memanfaatkan kanal radar.

**Mengapa patch lebih besar dari area peringatan?** Penyebab tidak hanya terlihat dari
bentuk area yang berubah, tetapi juga dari konteks sekitarnya (jalan di dekatnya, sungai,
lahan pertanian tetangga).

## 2. Struktur repo

```
.
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── synthetic.py   # generator data sintetis (bentuk + tanda spektral/radar per kelas)
    ├── make_data.py   # buat dan simpan dataset .npz
    ├── model.py       # arsitektur CNN (PyTorch)
    └── train.py       # latih, validasi, evaluasi, simpan confusion matrix
```

## 3. Cara menjalankan

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Buat data sintetis (7 kelas x 300 sampel)
python -m src.make_data --n-per-class 300 --out data/synthetic.npz

# 2) Latih dan evaluasi
python -m src.train --data data/synthetic.npz --epochs 15
```

Keluaran di folder `outputs/`:
`best_model.pt`, `classification_report.txt`, `confusion_matrix.png`, `history.json`.

GPU dipakai otomatis jika tersedia; CPU cukup untuk data sintetis ini.

## 4. Memakai data nyata

Kode training tidak peduli asal datanya selama formatnya cocok. File `.npz` berisi:

- `X`: `float32`, bentuk `(N, C, H, W)` untuk patch Sentinel-1/2 + mask peringatan (mask di **kanal terakhir**)
- `y`: `int64`, bentuk `(N,)` untuk indeks kelas
- `classes` (opsional): nama kelas

Langkah umum untuk data nyata:

1. **Peringatan**: unduh dari sumber terbuka seperti Global Forest Watch (mis. peringatan
   berbasis radar Sentinel-1 dan optik Sentinel-2/Landsat).
2. **Citra**: ekspor komposit pasca-gangguan Sentinel-2 dan Sentinel-1 (mis. via Google
   Earth Engine atau Copernicus Data Space), dipotong per peringatan.
3. **Label**: kumpulkan contoh berlabel (interpretasi visual oleh ahli, data lapangan,
   atau peta konsesi/tambang). Ini bagian paling mahal dan paling menentukan mutu.
4. **Pembagian data**: pisahkan latih/uji **berdasarkan lokasi/wilayah** (bukan acak per
   sampel) agar tidak terjadi kebocoran spasial yang membuat akurasi terlihat terlalu bagus.
5. Sesuaikan `in_channels` bila jumlah kanal berbeda (otomatis dari bentuk `X`).

## 5. Keterbatasan (sejalan dengan materi)

- Model dapat keliru membedakan **kebakaran, banjir, dan gangguan alami** lain karena
  tampilannya mirip.
- Hasil bergantung pada **ketersediaan citra, awan, dan frekuensi pembaruan** (mis. bulanan).
- Citra hanya menangkap **penyebab langsung yang terlihat**, bukan faktor sosial-ekonomi
  di baliknya (harga komoditas, tata kelola, tenure lahan).
- **Data lapangan** tetap diperlukan untuk validasi dan perbaikan model.
- Data sintetis di repo ini jauh lebih bersih daripada kenyataan; jangan menyimpulkan
  performa nyata dari angkanya.

## 6. Ide pengembangan

- Tambahkan data deret waktu (sebelum/sesudah) dan fitur jarak ke jalan/sungai.
- Analisis **urutan penyebab** (mis. jalan -> penebangan -> pertanian).
- Kalibrasi probabilitas dan tandai prediksi berkeyakinan rendah untuk dicek manual.
- Bandingkan dengan arsitektur lain (ResNet kecil, U-Net untuk segmentasi piksel).

## Lisensi

Pilih lisensi sebelum publikasi (mis. MIT) dan tambahkan file `LICENSE`.
