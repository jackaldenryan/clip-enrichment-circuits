async function microscopeUrl(parts) {
  // Hacky way to fix url
  let add = "add";
  if ((parts[2] == 6 && parts[1] != 0) || parts[2] == 8) {
    add = "Add";
  }

  const layerName = `image_block_${parts[0]}_${parts[1]}_${add}_${parts[2]}_0`;

  const visUrl =
    "https://microscope.openai.com/api/encyclopedia_endpoints/call:FeatureVisEndpoints.op_feature_vis?args=" +
    encodeURIComponent(
      JSON.stringify([
        "FeatureVisEndpoints",
        [],
        "op_feature_vis",
        ["contrastive_rn50", layerName, "channel", 4096, 0],
      ])
    );

  let featureVisImageUrl = "";
  try {
    const visualizations = await fetch(
      "https://ancient-truth-52ed.potato.workers.dev/?" +
        encodeURIComponent(visUrl)
    ).then((res) => res.json());

    featureVisImageUrl =
      visualizations.result.channels[parts[3]].images[0].image.url;

    console.log(featureVisImageUrl);
  } catch (e) {}

  return {
    featureVis: featureVisImageUrl,
    url: `https://microscope.openai.com/models/contrastive_rn50/${layerName}/${parts[3]}`,
  };
}

async function processClipN(el) {
  const id = el.dataset.id;
  const parts = id.split("/");
  const { featureVis, url } = await microscopeUrl(parts);
  el.innerHTML += `
        <code class="clip-n"><a target="_blank" href="${url}">${id} <img src="${featureVis}" style="border-radius: 4px" height="14" alt=""></a></code>
    `;
}

(async () => {
  for (const el of document.querySelectorAll(".clip-explicit")) {
    el.addEventListener("click", () => {
      el.classList.remove("clip-explicit");
    });
  }

  await Promise.all(
    Array.prototype.map.call(document.querySelectorAll("clip-n"), (el) =>
      processClipN(el)
    )
  );
})();
