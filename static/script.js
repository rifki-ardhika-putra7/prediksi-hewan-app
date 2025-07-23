// frontend/script.js
async function predict() {
    const input = document.getElementById('imageInput');
    const file = input.files[0];
    if (!file) return alert("Pilih gambar dulu!");

    const formData = new FormData();
    formData.append("image", file);

    try {
        const res = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await res.json();

        // Tampilkan semua info termasuk deskripsi dan fun fact
        document.getElementById('result').innerHTML = `
            <div class="bg-white rounded-lg shadow p-4">
                <h2 class="text-xl font-bold mb-2">🐾 ${data.label}</h2>
                <p><strong>Confidence:</strong> ${data.confidence.toFixed(2)}%</p>
                <p><strong>Deskripsi:</strong> ${data.description}</p>
                <p><strong>Fakta Menarik:</strong> ${data.fun_fact}</p>
                <p><strong>Habitat:</strong> ${data.habitat}</p>
                <p><strong>Diet:</strong> ${data.diet}</p>
                <p><strong>Ukuran:</strong> ${data.size}</p>
                <p><strong>Status Konservasi:</strong> ${data.conservation_status}</p>
            </div>
        `;
    } catch (err) {
        console.error("Gagal memuat prediksi:", err);
        document.getElementById('result').innerText = "Terjadi kesalahan saat memproses gambar.";
    }
}

// Preview
document.getElementById('imageInput').addEventListener('change', function(e) {
    const reader = new FileReader();
    reader.onload = function(event) {
        document.getElementById('preview').src = event.target.result;
    };
    reader.readAsDataURL(e.target.files[0]);
});
