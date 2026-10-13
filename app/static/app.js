// Drag-drop + preview + confidence slider + spinner
document.addEventListener("DOMContentLoaded", () => {
  const dz = document.getElementById("dropzone");
  const fi = document.getElementById("file-input");
  const prev = document.getElementById("preview");
  const slider = document.getElementById("conf-slider");
  const val = document.getElementById("conf-val");
  const form = document.getElementById("inspect-form");
  const btn = document.getElementById("detect-btn");
  const spin = document.getElementById("spinner");
  const btnText = document.getElementById("btn-text");

  if (slider && val) {
    slider.addEventListener("input", () => {
      val.textContent = parseFloat(slider.value).toFixed(2);
    });
  }
  if (dz && fi) {
    dz.addEventListener("click", () => fi.click());
    ["dragover", "dragenter"].forEach((e) =>
      dz.addEventListener(e, (ev) => {
        ev.preventDefault();
        dz.classList.add("dragover");
      })
    );
    ["dragleave", "drop"].forEach((e) =>
      dz.addEventListener(e, (ev) => {
        ev.preventDefault();
        dz.classList.remove("dragover");
      })
    );
    dz.addEventListener("drop", (ev) => {
      if (ev.dataTransfer.files.length) {
        fi.files = ev.dataTransfer.files;
        showPreview(fi.files[0]);
      }
    });
    fi.addEventListener("change", () => {
      if (fi.files.length) showPreview(fi.files[0]);
    });
  }
  function showPreview(file) {
    if (!file || !file.type.startsWith("image/")) return;
    const url = URL.createObjectURL(file);
    if (prev) {
      prev.src = url;
      prev.hidden = false;
    }
  }
  if (form) {
    form.addEventListener("submit", () => {
      if (btn) btn.disabled = true;
      if (spin) spin.hidden = false;
      if (btnText) btnText.textContent = "Detecting...";
    });
  }
  // Colour dots per class on result page
  const colors = {
    missing_hole: "#dc2626",
    mouse_bite: "#ea580c",
    open_circuit: "#ca8a04",
    short: "#1d4ed8",
    spur: "#9333ea",
    spurious_copper: "#15803d",
  };
  document.querySelectorAll(".dot[data-cls]").forEach((el) => {
    const c = el.getAttribute("data-cls");
    if (colors[c]) el.style.background = colors[c];
  });
});
