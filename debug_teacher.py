import torch
import numpy as np
from pathlib import Path

from movie_edge_sim.data import load_ratings_auto, load_item_genres_auto
from movie_edge_sim.temporal_realworld import build_user_time_histories, build_realworld_temporal_dataset, chronological_train_val_split, grouped_indices_by_user, RealWorldTemporalEncoder, load_compatible_temporal_state
from movie_edge_sim.novel_realworld_env import NovelRealWorldCachingEnv, RealWorldEnvConfig
from movie_edge_sim.novel_graph_policy import TemporalGraphCooperativePolicy, logits_to_cache_items
from compare_related_work_papers import _adapt_node_features, _adapt_candidate_features

def test_decoder():
    device = "cpu"
    dataset_dir = Path("data/ml-100k")
    ratings = load_ratings_auto(dataset_dir)
    histories = build_user_time_histories(ratings)
    
    temporal_model = RealWorldTemporalEncoder(
        num_items=max(max(h.items) for h in histories.values()),
        num_users=max(histories.keys()),
        window_size=8,
        embed_dim=32,
        hidden_dim=64,
        num_heads=2,
    )
    
    env_cfg = RealWorldEnvConfig(
        n_sbs=8,
        n_ues=60,
        cache_capacity=10,
        fp=30,
        grid_size=200.0,
        seed=42,
    )
    env = NovelRealWorldCachingEnv(env_cfg, temporal_model, histories)
    obs = env.reset(seed=3000)
    
    model = TemporalGraphCooperativePolicy(
        node_feat_dim=137,
        candidate_feat_dim=74,
        hidden_dim=64,
        fp=30,
    )
    
    model_path = Path("outputs/fast_100k_runs/latest/novel_realworld_main/temporal_graph_policy.pt")
    if model_path.exists():
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
    
    teacher_items = env.cooperative_teacher_action()
    teacher_scores = env.cooperative_teacher_scores()
    
    node = _adapt_node_features(obs["node_features"], 137, env.embed_dim)
    cand = _adapt_candidate_features(obs["candidate_features"], 74, env.embed_dim)
    node_t = torch.as_tensor(node, dtype=torch.float32)
    cand_t = torch.as_tensor(cand, dtype=torch.float32)
    adj_t = torch.as_tensor(obs["adjacency"], dtype=torch.float32)
    mask_t = torch.as_tensor(obs["action_mask"], dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        logits = model(node_t, cand_t, adj_t, mask_t)
        
    chosen = logits_to_cache_items(
        logits, 
        env, 
        diversity_penalty=0.35, 
        teacher_scores=teacher_scores, 
        teacher_guidance_weight=1.5
    )
    
    print("\n[Node 0 Debug]")
    print(f"Teacher Items Selection: {teacher_items[0]}")
    print(f"GNN Decoder Selection: {chosen[0]}")
    
    overlap = len(set(teacher_items[0]) & set(chosen[0]))
    print(f"Overlap: {overlap} out of 10")
    
if __name__ == "__main__":
    test_decoder()
