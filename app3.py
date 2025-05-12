import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import pandas as pd

st.header("📊 시뮬레이션 - 유사 유저 추천기")
st.write("장비 세팅을 입력하면, 딥러닝 모델을 통해 유사한 고레벨 유저를 추천합니다.")

# ------------------------------
# 모델 정의 (Deepsets 구조)
class DeepMaskedModel(nn.Module):
    def __init__(self, embedding_info, num_cont_features, emb_dim=8):
        super().__init__()
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(cardinality, emb_dim)
            for name, cardinality in embedding_info.items()
        })
        total_emb_dim = emb_dim * len(embedding_info)
        self.input_dim = total_emb_dim + num_cont_features
        self.phi = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.rho = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_cat, x_cont, mask):
        emb = [self.embeddings[name](x_cat[:, :, i]) for i, name in enumerate(self.embeddings)]
        emb = torch.cat(emb, dim=-1)
        x = torch.cat([emb, x_cont], dim=-1)
        encoded = self.phi(x)
        masked = encoded * mask.unsqueeze(-1)
        pooled = masked.sum(dim=1)
        return pooled

# ------------------------------
# 장비 인코딩 함수 (연속형 19차원 벡터만 추출)
def encode_item_to_cont_vector(item):
    vec = [
        item['boss_dmg'], item['ignore_def'], item['all_stat_total'], item['damage'],
        0, 0, item['all_stat_total'], item['starforce'], 0,
        0, item['mainstat_total'], item['power_total'], 0, 0,
        0, 0, 0, 0, item['item_count']
    ]
    return vec

# ------------------------------
# 전역 embedding 정보 정의
embedding_info = {
    'subclass': 46,
    'equipment_slot': 24,
    'main_stat_type': 6,
    'item_group': 15,
    'starforce_scroll_flag': 2,
    'potential_option_grade': 6,
    'additional_potential_option_grade': 6,
    'main_pot_grade_summary': 42,
    'add_pot_grade_summary': 55,
    'potential_status': 27
}

# ------------------------------
@st.cache_resource
def load_model():
    model = DeepMaskedModel(embedding_info, num_cont_features=19)
    model.load_state_dict(torch.load("best_model_r2_0.7083_rmse_0.70.pt", map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_high_level_user_vectors():
    vecs = np.load("high_level_user_vectors.npy")
    nicks = np.load("high_level_user_nicks.npy", allow_pickle=True)
    return vecs, nicks

@st.cache_resource
def load_job_avg_vectors():
    with open("subclass_profiles.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_user_job_map():
    df_stat = pd.read_csv("stat_merged.csv")  # 수정 필요 시 경로 변경
    df_stat = df_stat[['nickname', 'subclass']].drop_duplicates()
    return dict(zip(df_stat['nickname'], df_stat['subclass']))

model = load_model()
hl_vecs, hl_nicks = load_high_level_user_vectors()
job_avg_vectors = load_job_avg_vectors()
user_job_map = load_user_job_map()

# ------------------------------
# 사용자 입력 받기
job_list = sorted(job_avg_vectors.keys())
user_job = st.selectbox("직업을 선택하세요:", job_list)
top_slots = st.multiselect("Top-N 장비 부위를 선택하세요:", ['무기', '모자', '장갑', '신발', '망토', '상의', '하의'], default=['무기', '모자', '장갑'])

user_input = {}
for part in top_slots:
    with st.expander(f"{part} 정보 입력"):
        item = {}
        item['item_group'] = st.selectbox(f"{part} - 장비 세트", ['파프니르', '앱솔랩스', '도전자'], key=part)
        item['starforce'] = st.number_input(f"{part} - 스타포스", 0, 25, 15, key=part+"sf")
        item['mainstat_total'] = st.number_input(f"{part} - 주스탯 합", 0, 9999, 100, key=part+"main")
        item['power_total'] = st.number_input(f"{part} - 공격력/마력 합", 0, 999, 80, key=part+"pow")
        item['all_stat_total'] = st.number_input(f"{part} - 올스탯 합", 0, 99, 0, key=part+"all")
        item['potential_option_grade'] = st.selectbox(f"{part} - 잠재옵션 등급", ['레전드리', '유니크', '에픽', '레어'], key=part+"p1")
        item['additional_potential_option_grade'] = st.selectbox(f"{part} - 에디셔널 등급", ['레전드리', '유니크', '에픽', '레어'], key=part+"p2")
        for i in range(1, 4):
            item[f'potential_option_{i}_grade'] = st.selectbox(f"{part} - 잠재옵션 {i}", ['S', 'A', 'B', '기타'], key=f"{part}po{i}")
        for i in range(1, 4):
            item[f'additional_potential_option_{i}_grade'] = st.selectbox(f"{part} - 에디셔널 옵션 {i}", ['S', 'A', 'B', '기타'], key=f"{part}apo{i}")
        item['boss_dmg'] = st.slider(f"{part} - 보스 데미지 총합", 0.0, 100.0, 30.0, key=part+"bd")
        item['ignore_def'] = st.slider(f"{part} - 방무 총합", 0.0, 100.0, 20.0, key=part+"id")
        item['damage'] = st.slider(f"{part} - 데미지 총합", 0.0, 100.0, 25.0, key=part+"dg")
        item['item_count'] = 1
        user_input[part] = item

# ------------------------------
# 유사 유저 추천
if st.button("🔍 유사 유저 추천"):
    cont_vecs = [encode_item_to_cont_vector(user_input[part]) for part in top_slots]
    avg_vec = np.mean(cont_vecs, axis=0)

    x_cont = torch.tensor(avg_vec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    x_cat = torch.zeros((1, 1, len(embedding_info)), dtype=torch.long)
    mask = torch.tensor([[1.0]])

    with torch.no_grad():
        user_vec = model(x_cat, x_cont, mask).cpu().numpy().reshape(1, -1)

    # ✅ 선택된 subclass에 해당하는 유저만 필터링
    subclass_filtered_idx = [
        i for i, nick in enumerate(hl_nicks)
        if user_job_map.get(nick) == user_job
    ]
    filtered_vecs = hl_vecs[subclass_filtered_idx]
    filtered_nicks = hl_nicks[subclass_filtered_idx]

    if len(filtered_vecs) == 0:
        st.warning("해당 직업의 추천 유저가 존재하지 않습니다.")
    else:
        similarities = cosine_similarity(user_vec, filtered_vecs)[0]
        top5_idx = np.argsort(similarities)[::-1][:5]

        st.subheader("🔗 추천 유사 유저 Top-5")
        for idx in top5_idx:
            st.write(f"✅ {filtered_nicks[idx]} (유사도: {similarities[idx]:.3f})")
