# K/S 和反射率数据源调研

## 数据需求

我们需要收集：
- **K/S 数据**：吸收系数 K 和散射系数 S（31个波长点，400-700nm）
- **反射率数据**：R(λ)（可以用于反推 K/S）
- **材料信息**：颜料名称、化学组成、厂商、批次等

目标：**200-500 种真实材料**

---

## 一、公开数据源（免费）

### 1.1 CIE 数据库

**网站**：https://cie.co.at/data-tables

**可获取数据**：
- ✅ CIE 标准光源光谱（D65, A, F等）
- ✅ 观察者函数（1931 2°, 1964 10°）
- ⚠️ **反射率数据有限**（主要是标准色卡）
- ❌ **K/S 数据很少**

**估计材料数量**：~20 种标准色卡

**访问方式**：
```python
# 可以直接下载 CSV/Excel 文件
# 或者使用 Python 包
pip install colour-science
```

**数据质量**：★★★★★（权威标准）

---

### 1.2 Munsell 色卡数据

**网站**：
- Rochester Institute of Technology (RIT): https://www.rit.edu/science/munsell-color-science-lab-educational-resources
- Munsell Renotation Data: https://www.munsellcolourscienceforpainters.com/

**可获取数据**：
- ✅ Munsell 色卡的反射率数据（~1600 个色块）
- ✅ 完整的光谱数据（380-780nm）
- ❌ **不是 K/S，是反射率**（需要反推）

**文件格式**：通常是 `.txt` 或 `.csv`

**示例数据**：
```
# Munsell 2.5R 6/10 的反射率
Wavelength(nm)  Reflectance
400             0.052
410             0.058
420             0.065
...
700             0.682
```

**反推 K/S 的方法**：
```python
from scipy.optimize import minimize

def reflectance_to_ks(R):
    """
    从反射率 R 反推 K/S

    Kubelka-Munk 公式：
    K/S = (1 - R)² / (2R)
    """
    # 简化方法（假设 S 为常数）
    ks_ratio = (1 - R)**2 / (2 * R)

    # 如果需要分离 K 和 S，需要额外信息或假设
    # 例如：假设 S = 某个常数（基于经验）
    S_assumed = 1.0
    K = ks_ratio * S_assumed

    return K, S_assumed * np.ones_like(K)

# 或者优化方法（如果有多个浓度的测量）
def optimize_ks_from_multi_concentration(R_list, conc_list):
    """
    从多个浓度的反射率反推 K 和 S
    """
    def objective(params):
        K, S = params[:31], params[31:]
        loss = 0
        for R_measured, c in zip(R_list, conc_list):
            K_mix = K * c
            S_mix = S * c
            R_pred = 1 + K_mix/S_mix - np.sqrt((K_mix/S_mix)**2 + 2*K_mix/S_mix)
            loss += np.sum((R_pred - R_measured)**2)
        return loss

    result = minimize(objective, x0=np.ones(62))
    return result.x[:31], result.x[31:]
```

**估计材料数量**：~1600 个色块（但不是独立材料，是混合色）

**数据质量**：★★★★☆（标准色卡，但需要反推 K/S）

---

### 1.3 学术论文数据集

**来源**：
1. **Color Research & Application** (Wiley)
2. **Journal of the Optical Society of America A**
3. **Applied Optics**

**搜索关键词**：
- "Kubelka-Munk"
- "K/S ratio"
- "pigment reflectance"
- "spectral data"
- "color matching"

**具体数据集**：

#### a) 纺织染料数据
**论文**：Shen et al., "Color Recipe Prediction" (多篇)
- 可能包含几十种染料的 K/S 数据
- 需要联系作者或查找补充材料

#### b) 涂料颜料数据
**论文**：Paint & Coatings Industry 期刊
- 部分论文包含颜料光谱数据

#### c) 食品色素数据
**论文**：Food Chemistry 期刊
- 食品级色素的反射率/K/S

**访问方式**：
```bash
# 1. 通过学术搜索
Google Scholar: "Kubelka-Munk pigment data"
Web of Science: TS=("K/S ratio" AND pigment)

# 2. 下载补充材料
# 很多论文的 Supplementary Materials 包含完整数据

# 3. 联系作者
# 发邮件请求原始数据
```

**估计材料数量**：~50-100 种（需要逐篇整理）

**数据质量**：★★★★☆（学术标准，但分散）

---

### 1.4 国家标准数据库

**中国**：
- GB 标准：部分包含颜料光谱数据
- 网站：https://openstd.samr.gov.cn/

**美国**：
- ASTM Standards: https://www.astm.org/
- ASTM D2805 (Standard Practice for Computerized Color Matching)
- 部分标准包含参考数据

**欧洲**：
- ISO Standards: https://www.iso.org/
- ISO 11664 系列（色度学）

**可获取数据**：
- ⚠️ 通常只有**测量方法**，不直接提供数据
- 但标准附录可能有参考样本数据

**估计材料数量**：~10-20 种参考样本

**数据质量**：★★★★★（权威）

---

## 二、商业数据库（付费/合作）

### 2.1 X-Rite 数据库

**公司**：X-Rite (Pantone 母公司)
**产品**：
- **PantoneLIVE**：数字色彩平台
- **Color Master**：配色软件

**数据内容**：
- ✅ 数千种 Pantone 色彩的光谱数据
- ✅ 常见颜料的 K/S 数据
- ✅ 配方数据库

**访问方式**：
1. **购买软件**：Color Master (~$5000-$15000)
2. **订阅 PantoneLIVE**：~$1000/年
3. **学术合作**：联系 X-Rite 教育部门

**数据导出**：
- 可以导出为 CSV/Excel
- 需要注意许可协议（可能限制再分发）

**估计材料数量**：~500-1000 种

**数据质量**：★★★★★（行业标准）

**成本**：$$$$（高）

---

### 2.2 Datacolor 数据库

**公司**：Datacolor
**产品**：
- **MATCH Textile**：纺织配色
- **MATCH Pigment**：涂料配色

**数据内容**：
- ✅ 染料和颜料 K/S 数据
- ✅ 配方数据库
- ✅ 质量控制数据

**访问方式**：
- 购买软件：~$10000-$30000
- 或行业合作

**估计材料数量**：~500-800 种

**数据质量**：★★★★★

**成本**：$$$$（高）

---

### 2.3 颜料厂商技术手册

**主要厂商**：

#### a) DuPont（杜邦）
- TiO₂ 系列（钛白粉）
- 网站：https://www.dupont.com/
- **技术手册**：部分包含反射率曲线
- **获取**：销售代表、技术支持

#### b) BASF（巴斯夫）
- 广泛的有机和无机颜料
- 网站：https://www.basf.com/
- **Pigment Finder**：在线数据库
- **技术数据表 (TDS)**：包含部分光谱信息

#### c) Clariant（科莱恩）
- 特殊效果颜料
- 网站：https://www.clariant.com/
- **技术文档**：可申请

#### d) Sun Chemical（太阳化工）
- 印刷油墨颜料
- 网站：https://www.sunchemical.com/

**访问方式**：
```python
# 1. 下载技术数据表（TDS）
# 网站上搜索产品名称 + "TDS"

# 2. 联系技术支持
# 发邮件说明研究目的，请求光谱数据

# 3. 购买样品
# 部分厂商愿意提供样品 + 详细数据
```

**估计材料数量**：~100-200 种（需要逐家收集）

**数据质量**：★★★★★（官方数据）

**成本**：$-$$（免费到低成本）

---

## 三、自行测量

### 3.1 分光光度计

**设备选择**：

| 型号 | 价格 | 波长范围 | 精度 | 推荐度 |
|------|------|---------|------|--------|
| **Konica Minolta CM-5** | ~$5000 | 400-700nm | ±0.05 ΔE | ★★★★★ |
| **X-Rite Ci7800** | ~$15000 | 360-750nm | ±0.03 ΔE | ★★★★★ |
| **BYK-Gardner** | ~$3000-8000 | 400-700nm | ±0.1 ΔE | ★★★★☆ |
| **3nh NH300** | ~$500-1000 | 400-700nm | ±0.3 ΔE | ★★★☆☆ |

**推荐**（性价比）：
- **Konica Minolta CM-5**：科研标准，价格适中
- **3nh NH300**：预算有限时的选择

**测量流程**：
```python
# 1. 样品制备
# - 购买颜料粉末（~$10-50/种）
# - 制备标准样板（涂布在标准卡纸上）
# - 确保厚度均匀（避免透明效应）

# 2. 仪器校准
# - 白板校准
# - 黑板校准

# 3. 测量
# - 每个样品测量 3-5 次
# - 取平均值

# 4. 数据记录
# - 导出为 CSV（通常仪器自带软件）
# - 格式：波长, 反射率

# 5. K/S 计算
# - 使用 Kubelka-Munk 公式
# - 或仪器软件直接输出
```

**颜料采购**：
```
来源：
1. 化学试剂公司（Sigma-Aldrich, Aladdin）
   - 高纯度，标准化
   - 价格：~$20-100/种

2. 颜料供应商（阿里巴巴）
   - 工业级，价格低
   - 价格：~$5-20/种
   - ⚠️ 批次差异较大

3. 美术用品店
   - 艺术级颜料
   - 价格：~$10-50/种

推荐：混合采购
- 核心材料（20种）：高纯度试剂
- 常用材料（80种）：工业级
- 补充材料（100种）：混合来源
```

**成本估算**：
```
设备：$3000-5000（分光光度计）
颜料：$1000-2000（200种 × $10 均价）
耗材：$500（卡纸、溶剂等）
人工：~2-3个月

总计：~$5000-8000 + 人力
```

**估计材料数量**：**200+ 种**（完全可控）

**数据质量**：★★★★★（自己控制标准）

---

## 四、开源社区

### 4.1 GitHub 数据集

**搜索**：
```bash
GitHub: "pigment spectral data"
GitHub: "Kubelka-Munk dataset"
GitHub: "color matching data"
```

**已知项目**：

#### a) colour-datasets
- 网址：https://github.com/colour-science/colour-datasets
- **内容**：各种色彩科学数据集
- **包含**：部分标准色卡的反射率数据
- **格式**：Python 包，易于使用

```python
pip install colour-datasets

from colour_datasets import load_dataset

# 加载 Munsell 数据
munsell = load_dataset('Munsell Renotation')
```

#### b) PaperStuff
- 一些研究者会在 GitHub 分享论文数据
- 搜索：配色相关论文的 GitHub repo

**估计材料数量**：~50-100 种

**数据质量**：★★★★☆（取决于来源）

---

### 4.2 建立开源数据库（长期）

**方案**：
```markdown
# PigmentDB - 开源颜料光谱数据库

## 愿景
建立一个开放的、社区驱动的颜料 K/S 数据库

## 实施
1. 创建 GitHub 仓库
2. 定义数据标准（JSON/CSV 格式）
3. 鼓励用户贡献数据
4. 数据验证和质量控制

## 激励机制
- 贡献者署名
- 下载需要贡献（1:1 交换）
- 学术引用

## 许可
MIT License 或 CC-BY-4.0
```

**潜力**：如果成功，可能积累 **500-1000+ 种**数据

---

## 五、实际推荐方案

### 阶段 1：快速启动（1个月，~100种）

```
数据源优先级：

1. Munsell 色卡数据（免费）
   - 下载 Munsell Renotation 数据
   - 反推 K/S
   - 数量：~1600 色块（虽然是混合色，但可作为初步验证）

2. CIE 数据库（免费）
   - 下载标准色卡
   - 数量：~20 种

3. 学术论文数据（免费，需要整理）
   - 搜索 5-10 篇相关论文
   - 提取补充材料
   - 数量：~30-50 种

4. 颜料厂商技术手册（免费-低成本）
   - BASF Pigment Finder
   - DuPont TiO₂ 数据
   - 数量：~20-30 种

总计：~100 种真实材料 + Munsell 扩展
```

### 阶段 2：自行测量（2-3个月，+100种）

```
采购：
1. 分光光度计（如 Konica Minolta CM-5）
   - 成本：~$5000
   - 可以测量无限种材料

2. 颜料样品
   - 从试剂公司购买 50 种高纯度
   - 从工业供应商购买 50 种常用
   - 成本：~$1000-2000

总计：+100 种高质量自测数据
```

### 阶段 3：行业合作（6-12个月，+200-300种）

```
合作对象：
1. 颜料厂商（互惠）
   - 提供技术支持换取数据
   - 帮助他们优化配方

2. 涂料/印刷公司
   - 联合研发
   - 共建数据库

3. 学术机构
   - 合作论文
   - 共享数据

总计：+200-300 种
```

---

## 六、数据标准化

### 6.1 统一格式

**推荐格式**（JSON）：
```json
{
  "material_id": "TiO2-R900",
  "name": "Titanium Dioxide Rutile",
  "category": "inorganic_pigment",
  "manufacturer": "DuPont",
  "batch": "2023-Q1",
  "wavelengths": [400, 410, 420, ..., 700],
  "K": [0.05, 0.06, 0.07, ..., 0.12],
  "S": [1.20, 1.18, 1.15, ..., 0.95],
  "reflectance": [0.95, 0.96, 0.96, ..., 0.93],
  "measurement_conditions": {
    "illuminant": "D65",
    "observer": "CIE1931_2deg",
    "geometry": "d/8",
    "instrument": "Konica Minolta CM-5"
  },
  "metadata": {
    "CAS": "13463-67-7",
    "color_index": "PW6",
    "particle_size_um": 0.3,
    "measurement_date": "2024-01-15"
  }
}
```

### 6.2 数据验证

```python
def validate_ks_data(data):
    """验证 K/S 数据的物理合理性"""

    # 1. 非负性
    assert np.all(data['K'] >= 0), "K must be non-negative"
    assert np.all(data['S'] >= 0), "S must be non-negative"

    # 2. 波长范围
    assert len(data['wavelengths']) == 31, "Must have 31 wavelength points"
    assert data['wavelengths'][0] == 400, "Start at 400nm"
    assert data['wavelengths'][-1] == 700, "End at 700nm"

    # 3. 光谱平滑性
    K_diff = np.diff(data['K'])
    assert np.all(np.abs(K_diff) < 0.5), "K spectrum too noisy"

    # 4. 反射率一致性（如果有）
    if 'reflectance' in data:
        K, S = np.array(data['K']), np.array(data['S'])
        R_calc = 1 + K/S - np.sqrt((K/S)**2 + 2*K/S)
        R_meas = np.array(data['reflectance'])
        error = np.mean(np.abs(R_calc - R_meas))
        assert error < 0.05, f"R mismatch too large: {error}"

    return True
```

---

## 七、总结

### 可行性评估

| 数据源 | 数量 | 成本 | 时间 | 质量 | 推荐 |
|--------|------|------|------|------|------|
| Munsell 数据 | ~1600 | 免费 | 1天 | ★★★★☆ | ✅ |
| CIE 数据库 | ~20 | 免费 | 1天 | ★★★★★ | ✅ |
| 学术论文 | ~50-100 | 免费 | 2周 | ★★★★☆ | ✅ |
| 颜料厂商 | ~100-200 | 低 | 1-2月 | ★★★★★ | ✅ |
| 自行测量 | 无限 | $5-8K | 2-3月 | ★★★★★ | ✅ |
| 商业数据库 | ~500-1000 | $$$$ | 即时 | ★★★★★ | ⚠️ |

### MVP 推荐方案（3个月，200种材料）

```
Week 1-2: 公开数据收集
- Munsell 数据（免费）
- CIE 数据库（免费）
- 学术论文（免费）
- 小计：~100 种基础数据

Week 3-4: 采购设备和样品
- 购买分光光度计（$5000）
- 购买 50 种颜料样品（$1000）

Week 5-12: 自行测量
- 测量 50 种颜料
- 扩展到 100-150 种
- 数据验证和标准化

总计：~200 种高质量数据 ✅
总成本：~$6000-8000
```

**结论**：**200种材料的MVP完全可行**，数据来源充足！
