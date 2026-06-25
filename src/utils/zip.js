export async function downloadZip(name, items) {
  const { default: JSZip } = await import("jszip");
  const zip = new JSZip();
  const folder = zip.folder(name);
  for (const item of items) {
    folder.file(item.file.name, item.file);
  }
  const blob = await zip.generateAsync({ type: "blob" });
  const href = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = href;
  anchor.download = `${name}.zip`;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(href), 0);
}
