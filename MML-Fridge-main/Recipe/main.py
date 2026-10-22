from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import sqlite3
import json
import os
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_FILE = "smart_fridge.db"

class RecommendRequest(BaseModel):
    selected_ingredients: List[str]
    all_fridge_ingredients: List[str]

# --- 新增：常用调料黑名单 ---
# 这里的食材在计算“匹配度”时会被忽略，不会拉低分数
CONDIMENTS = [
    'salt', 'pepper', 'water', 'oil', 'sugar', 'butter', 'olive oil', 
    'garlic', 'onion', 'soy sauce', 'vinegar', 'flour', 'broth', 'milk', 'spices'
]


@app.post("/api/recommend")
def get_recommendations(req: RecommendRequest):
    if not req.selected_ingredients:
        return []

    lower_selected = [ing.lower() for ing in req.selected_ingredients]
    lower_all_fridge = [ing.lower() for ing in req.all_fridge_ingredients]

    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 1. 扩大范围：把沾边的全抓出来
    conditions = " OR ".join([f"name LIKE ?" for _ in lower_selected])
    params = [f"%{ing}%" for ing in lower_selected]
    query = f"SELECT recipe_id FROM recipe_ingredients WHERE {conditions} GROUP BY recipe_id LIMIT 200"
    cursor.execute(query, params)
    valid_recipe_ids = [row['recipe_id'] for row in cursor.fetchall()]

    if not valid_recipe_ids:
        return []

    results = []
    for rid in valid_recipe_ids:
        cursor.execute("SELECT * FROM recipes WHERE id = ?", (rid,))
        recipe_meta = cursor.fetchone()
        
        cursor.execute("SELECT name, amount FROM recipe_ingredients WHERE recipe_id = ?", (rid,))
        needed_ingredients = cursor.fetchall()
        
        matched_count = 0
        selected_hits = 0 # 核心指标：记录这道菜究竟命中了几个你【选中的食材】
        ingredients_list = []
        
        for ing in needed_ingredients:
            ing_name = ing['name'].lower()
            # 调料黑名单
            is_condiment = any(c in ing_name for c in ['salt', 'pepper', 'water', 'oil', 'sugar', 'butter', 'garlic', 'onion', 'soy sauce', 'vinegar', 'spices', 'broth'])
            
            have_it = False
            # 检查冰箱里到底有没有（使用正则精准匹配，防止 'egg' 匹配到 'eggplant'）
            for fridge_item in lower_all_fridge:
                base_word = fridge_item.rstrip('s') 
                if re.search(rf'\b{re.escape(base_word)}\b', ing_name):
                    have_it = True
                    break
            
            # 检查这个食材是不是你当前选中的？
            for sel_item in lower_selected:
                base_sel = sel_item.rstrip('s')
                if re.search(rf'\b{re.escape(base_sel)}\b', ing_name):
                    selected_hits += 1
                    break
            
            # 算分逻辑：如果你有，或者是调料，分子加一
            if have_it or is_condiment:
                matched_count += 1
                    
            ingredients_list.append({
                "name": ing['name'],
                "amount": ing['amount'],
                "have": have_it # 修复打勾Bug：只有冰箱里真有的，才给UI返回True
            })
            
        # 防止除以0
        match_score = int((matched_count / len(needed_ingredients)) * 100) if len(needed_ingredients) > 0 else 100
        
        # 过滤掉那些完全没命中你选中食材的噪音菜谱
        if selected_hits > 0:
            results.append({
                "name": recipe_meta['name'],
                "difficulty": recipe_meta['difficulty'],
                "time": recipe_meta['time'],
                "calories": recipe_meta['calories'],
                "desc": recipe_meta['desc'],
                "match": match_score,
                "selected_hits": selected_hits, # 保存这个关键指标用于排序
                "ingredients": ingredients_list,
                "steps": json.loads(recipe_meta['steps'])
            })

    conn.close()
    
    # 【终极排序逻辑】
    # 第一关键字：命中选中的食材数量 (降序) -> 确保西红柿和鸡蛋都在的排第一
    # 第二关键字：整体匹配分数 (降序) -> 在同样包含西红柿鸡蛋的前提下，缺的配料越少越靠前
    results.sort(key=lambda x: (x['selected_hits'], x['match']), reverse=True)
    return results[:15]