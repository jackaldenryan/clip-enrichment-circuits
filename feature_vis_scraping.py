import urllib.request, json

with urllib.request.urlopen("https://ggoh-staging-dot-encyclopedia-251300.wl.r.appspot.com/api/encyclopedia_endpoints/call:FeatureVisEndpoints.op_feature_vis?args=[%22FeatureVisEndpoints%22%2C[]%2C%22op_feature_vis%22%2C[%22contrastive_rn50%22%2C%22image_block_4_1_Add_6_0%22%2C%22channel%22%2C4096%2C0]]&fbclid=IwAR0UA1fHHMVxmmCVQPZHbuWEh0uYvALrX2ewTw4VLYV3iHE8CEVttEC5w5I") as url:
    data = json.loads(url.read().decode())

    cnt = 0
    for chl in data["result"]["channels"]:
        print(cnt)
        img_url = chl["images"][0]["image"]["url"]
        urllib.request.urlretrieve(img_url, f"seri/datasets/scrape4_1_6/channel{cnt}.png")
        cnt += 1