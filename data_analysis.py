import pandas as pd
import pingouin as pg

# ---------- Step 1: 读取数据 ----------
file_path = r"D:\projects\AI & education\data.xlsx"
df = pd.read_excel(file_path)

# 确保 Time 和 Group 是字符串类型（混合 ANOVA 要求）
df['Time'] = df['Time'].astype(str)
df['Group'] = df['Group'].astype(str)

# ---------- Step 2: 执行混合方差分析 ----------
aov_cw = pg.mixed_anova(dv='CW', within='Time', between='Group', subject='ID', data=df)
aov_i = pg.mixed_anova(dv='I', within='Time', between='Group', subject='ID', data=df)


# ---------- Step 3: 格式化为APA期刊风格 ----------
def format_anova_output(df_pg):
    df_formatted = df_pg[['Source', 'DF1', 'DF2', 'F', 'p-unc', 'np2']].copy()
    df_formatted['df'] = df_formatted['DF1'].astype(int).astype(str) + ', ' + df_formatted['DF2'].astype(int).astype(
        str)
    df_formatted = df_formatted.rename(columns={
        'Source': 'Source',
        'F': 'F',
        'p-unc': 'p',
        'np2': 'η²p'
    })[['Source', 'df', 'F', 'p', 'η²p']]

    # 保留三位小数 + APA 风格 p 值处理
    df_formatted['F'] = df_formatted['F'].round(2)
    df_formatted['p'] = df_formatted['p'].apply(lambda x: '< .001' if x < 0.001 else f'{x:.3f}')
    df_formatted['η²p'] = df_formatted['η²p'].round(3)

    return df_formatted


cw_apa = format_anova_output(aov_cw)
i_apa = format_anova_output(aov_i)

# ---------- Step 4: 导出到 Excel ----------
output_path = r"D:\projects\AI & education\anova_apa.xlsx"
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    cw_apa.to_excel(writer, sheet_name='CW_ANOVA_APA', index=False)
    i_apa.to_excel(writer, sheet_name='I_ANOVA_APA', index=False)


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2, shapiro
import pingouin as pg

# Step 1: 读取数据并格式转换
df = pd.read_excel(r"D:\projects\AI & education\data.xlsx")
df['ID'] = df['ID'].astype(str)
df['Time'] = df['Time'].astype(str)
df['Group'] = df['Group'].astype(str)
df['gender'] = df['gender'].astype(str)
df['hukou'] = df['hukou'].astype(str)

# Step 2: ICC 前置检验
icc_f = pg.intraclass_corr(data=df, targets='ID', raters='Time', ratings='F', nan_policy='omit')
icc_e = pg.intraclass_corr(data=df, targets='ID', raters='Time', ratings='E', nan_policy='omit')
icc_cw = pg.intraclass_corr(data=df, targets='ID', raters='Time', ratings='CW', nan_policy='omit')

print("📌 ICC for F:")
print(icc_f[["Type", "ICC", "CI95%"]])
print("\n📌 ICC for E:")
print(icc_e[["Type", "ICC", "CI95%"]])
print("\n📌 ICC for CW:")
print(icc_cw[["Type", "ICC", "CI95%"]])

# Step 3: MLM中介路径建模
# 路径 a：E ~ F
model_a = smf.mixedlm("E ~ F + Group + Time + gender + hukou", df, groups=df["ID"]).fit()
print("\n📈 Path a (F → E):")
print(model_a.summary())

# 路径 b+c′：CW ~ F + E
model_b = smf.mixedlm("CW ~ F + E + Group + Time + gender + hukou", df, groups=df["ID"]).fit()
print("\n📈 Path b+c′ (E → CW, controlling F):")
print(model_b.summary())

# 路径 c（总效应）：CW ~ F
model_c = smf.mixedlm("CW ~ F + Group + Time + gender + hukou", df, groups=df["ID"]).fit()
print("\n📈 Path c (F → CW):")
print(model_c.summary())

# Step 4: Monte Carlo估计间接效应 a × b
np.random.seed(42)
n_sim = 5000
a_mean = model_a.params["F"]
a_se = model_a.bse["F"]
b_mean = model_b.params["E"]
b_se = model_b.bse["E"]

a_sim = np.random.normal(a_mean, a_se, n_sim)
b_sim = np.random.normal(b_mean, b_se, n_sim)
ab_sim = a_sim * b_sim
ab_mean = np.mean(ab_sim)
ci_lower = np.percentile(ab_sim, 2.5)
ci_upper = np.percentile(ab_sim, 97.5)

print("\n📊 Monte Carlo Estimate of Indirect Effect:")
print(f"Indirect Effect = {ab_mean:.4f}")
print(f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")



# Step 5: 模型比较（完整 vs 空模型）
null_model = smf.mixedlm("CW ~ 1", df, groups=df["ID"]).fit()
ll_null = null_model.llf
ll_full = model_b.llf
df_null = null_model.model.exog.shape[1]
df_full = model_b.model.exog.shape[1]
df_diff = df_full - df_null
lr_stat = 2 * (ll_full - ll_null)
p_val_lr = chi2.sf(lr_stat, df_diff)

print("\n📌 Likelihood Ratio Test (Model Comparison)")
print(f"χ² = {lr_stat:.2f}, df = {df_diff}, p = {p_val_lr:.4f}")
if p_val_lr < 0.05:
    print("✅ 完整模型显著优于空模型，支持使用MLM")
else:
    print("⚠️ 模型改进不显著，MLM未必优于OLS")


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import chi2, shapiro
import pingouin as pg

# Step 1: 读取数据并格式转换
df = pd.read_excel(r"D:\projects\AI & education\data.xlsx")
df['ID'] = df['ID'].astype(str)
df['Time'] = df['Time'].astype(str)
df['Group'] = df['Group'].astype(str)
df['gender'] = df['gender'].astype(str)
df['hukou'] = df['hukou'].astype(str)

# Step 2: ICC 前置检验
icc_f = pg.intraclass_corr(data=df, targets='ID', raters='Time', ratings='F', nan_policy='omit')
icc_e = pg.intraclass_corr(data=df, targets='ID', raters='Time', ratings='E', nan_policy='omit')
icc_cw = pg.intraclass_corr(data=df, targets='ID', raters='Time', ratings='CW', nan_policy='omit')

print("📌 ICC for F:")
print(icc_f[["Type", "ICC", "CI95%"]])
print("\n📌 ICC for E:")
print(icc_e[["Type", "ICC", "CI95%"]])
print("\n📌 ICC for CW:")
print(icc_cw[["Type", "ICC", "CI95%"]])

# Step 3: MLM中介路径建模
# 路径 a：E ~ F
model_a = smf.mixedlm("E ~ F + Group + Time + gender + hukou", df, groups=df["ID"]).fit()
print("\n📈 Path a (F → E):")
print(model_a.summary())

# 路径 b+c′：CW ~ F + E
model_b = smf.mixedlm("CW ~ F + E + Group + Time + gender + hukou", df, groups=df["ID"]).fit()
print("\n📈 Path b+c′ (E → CW, controlling F):")
print(model_b.summary())

# 路径 c（总效应）：CW ~ F
model_c = smf.mixedlm("CW ~ F + Group + Time + gender + hukou", df, groups=df["ID"]).fit()
print("\n📈 Path c (F → CW):")
print(model_c.summary())

# Step 4: Monte Carlo估计间接效应 a × b
np.random.seed(42)
n_sim = 5000
a_mean = model_a.params["F"]
a_se = model_a.bse["F"]
b_mean = model_b.params["E"]
b_se = model_b.bse["E"]

a_sim = np.random.normal(a_mean, a_se, n_sim)
b_sim = np.random.normal(b_mean, b_se, n_sim)
ab_sim = a_sim * b_sim
ab_mean = np.mean(ab_sim)
ci_lower = np.percentile(ab_sim, 2.5)
ci_upper = np.percentile(ab_sim, 97.5)

print("\n📊 Monte Carlo Estimate of Indirect Effect:")
print(f"Indirect Effect = {ab_mean:.4f}")
print(f"95% CI = [{ci_lower:.4f}, {ci_upper:.4f}]")



# Step 5: 模型比较（完整 vs 空模型）
null_model = smf.mixedlm("CW ~ 1", df, groups=df["ID"]).fit()
ll_null = null_model.llf
ll_full = model_b.llf
df_null = null_model.model.exog.shape[1]
df_full = model_b.model.exog.shape[1]
df_diff = df_full - df_null
lr_stat = 2 * (ll_full - ll_null)
p_val_lr = chi2.sf(lr_stat, df_diff)

print("\n📌 Likelihood Ratio Test (Model Comparison)")
print(f"χ² = {lr_stat:.2f}, df = {df_diff}, p = {p_val_lr:.4f}")
if p_val_lr < 0.05:
    print("✅ 完整模型显著优于空模型，支持使用MLM")
else:
    print("⚠️ 模型改进不显著，MLM未必优于OLS")

import pandas as pd
import statsmodels.formula.api as smf


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# 读取数据
df = pd.read_excel(r"D:\projects\AI & education\data.xlsx")

# 变量处理
df['ID'] = df['ID'].astype(str)
df['Time'] = df['Time'].astype(str)
df['Group'] = df['Group'].astype(str)
df['gender'] = df['gender'].astype(str)
df['hukou'] = df['hukou'].astype(str)

# 分组
df_group0 = df[df['Group'] == '0']  # peer friendship
df_group1 = df[df['Group'] == '1']  # AI friendship


# 定义函数：中介分析三步 + Monte Carlo
def mlm_mediation(df_subset):
    results = {}

    # Path a: F → E
    model_a = smf.mixedlm("E ~ F + Time + gender + hukou", df_subset, groups=df_subset["ID"]).fit()
    results['a (F → E)'] = {
        'B': round(model_a.params['F'], 3),
        'SE': round(model_a.bse['F'], 3),
        'z': round(model_a.tvalues['F'], 3),
        'p': round(model_a.pvalues['F'], 3),
        '95% CI': [round(model_a.conf_int().loc['F'][0], 3), round(model_a.conf_int().loc['F'][1], 3)]
    }

    # Path b+c': E, F → I
    model_b = smf.mixedlm("I ~ F + E + Time + gender + hukou", df_subset, groups=df_subset["ID"]).fit()
    results['b (E → I)'] = {
        'B': round(model_b.params['E'], 3),
        'SE': round(model_b.bse['E'], 3),
        'z': round(model_b.tvalues['E'], 3),
        'p': round(model_b.pvalues['E'], 3),
        '95% CI': [round(model_b.conf_int().loc['E'][0], 3), round(model_b.conf_int().loc['E'][1], 3)]
    }
    results['c′ (direct)'] = {
        'B': round(model_b.params['F'], 3),
        'SE': round(model_b.bse['F'], 3),
        'z': round(model_b.tvalues['F'], 3),
        'p': round(model_b.pvalues['F'], 3),
        '95% CI': [round(model_b.conf_int().loc['F'][0], 3), round(model_b.conf_int().loc['F'][1], 3)]
    }

    # Total effect path c: F → I
    model_c = smf.mixedlm("I ~ F + Time + gender + hukou", df_subset, groups=df_subset["ID"]).fit()
    results['c (total)'] = {
        'B': round(model_c.params['F'], 3),
        'SE': round(model_c.bse['F'], 3),
        'z': round(model_c.tvalues['F'], 3),
        'p': round(model_c.pvalues['F'], 3),
        '95% CI': [round(model_c.conf_int().loc['F'][0], 3), round(model_c.conf_int().loc['F'][1], 3)]
    }

    # Monte Carlo simulation for indirect effect a × b
    np.random.seed(42)
    n_sim = 20000
    a_sim = np.random.normal(results['a (F → E)']['B'], results['a (F → E)']['SE'], n_sim)
    b_sim = np.random.normal(results['b (E → I)']['B'], results['b (E → I)']['SE'], n_sim)
    ab_sim = a_sim * b_sim
    ab_mean = np.mean(ab_sim)
    ci_lower = np.percentile(ab_sim, 2.5)
    ci_upper = np.percentile(ab_sim, 97.5)

    results['Indirect (a×b)'] = {
        'B': round(ab_mean, 4),
        '95% CI': f"[{round(ci_lower, 4)}, {round(ci_upper, 4)}]"
    }

    return results


# 运行两组分析
results_group0 = mlm_mediation(df_group0)
results_group1 = mlm_mediation(df_group1)

# 汇总输出
df_results = pd.DataFrame([results_group0, results_group1],
                          index=["Peer Friendship (Group 0)", "AI Friendship (Group 1)"])
print("\n📊 Group-wise Multilevel Mediation Results")
print(df_results)

# 可选保存
df_results.to_excel(r"D:\projects\AI & education\groupwise_mediation_resultsI.xlsx")

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# 读取数据
df = pd.read_excel(r"D:\projects\AI & education\data.xlsx")

# 变量处理
df['ID'] = df['ID'].astype(str)
df['Time'] = df['Time'].astype(str)
df['Group'] = df['Group'].astype(str)
df['gender'] = df['gender'].astype(str)
df['hukou'] = df['hukou'].astype(str)

# 分组
df_group0 = df[df['Group'] == '0']  # peer friendship
df_group1 = df[df['Group'] == '1']  # AI friendship


# 定义函数：中介分析三步 + Monte Carlo
def mlm_mediation(df_subset):
    results = {}

    # Path a: F → E
    model_a = smf.mixedlm("E ~ F + Time + gender + hukou", df_subset, groups=df_subset["ID"]).fit()
    results['a (F → E)'] = {
        'B': round(model_a.params['F'], 3),
        'SE': round(model_a.bse['F'], 3),
        'z': round(model_a.tvalues['F'], 3),
        'p': round(model_a.pvalues['F'], 3),
        '95% CI': [round(model_a.conf_int().loc['F'][0], 3), round(model_a.conf_int().loc['F'][1], 3)]
    }

    # Path b+c': E, F → CW
    model_b = smf.mixedlm("CW ~ F + E + Time + gender + hukou", df_subset, groups=df_subset["ID"]).fit()
    results['b (E → CW)'] = {
        'B': round(model_b.params['E'], 3),
        'SE': round(model_b.bse['E'], 3),
        'z': round(model_b.tvalues['E'], 3),
        'p': round(model_b.pvalues['E'], 3),
        '95% CI': [round(model_b.conf_int().loc['E'][0], 3), round(model_b.conf_int().loc['E'][1], 3)]
    }
    results['c′ (direct)'] = {
        'B': round(model_b.params['F'], 3),
        'SE': round(model_b.bse['F'], 3),
        'z': round(model_b.tvalues['F'], 3),
        'p': round(model_b.pvalues['F'], 3),
        '95% CI': [round(model_b.conf_int().loc['F'][0], 3), round(model_b.conf_int().loc['F'][1], 3)]
    }

    # Total effect path c: F → CW
    model_c = smf.mixedlm("CW ~ F + Time + gender + hukou", df_subset, groups=df_subset["ID"]).fit()
    results['c (total)'] = {
        'B': round(model_c.params['F'], 3),
        'SE': round(model_c.bse['F'], 3),
        'z': round(model_c.tvalues['F'], 3),
        'p': round(model_c.pvalues['F'], 3),
        '95% CI': [round(model_c.conf_int().loc['F'][0], 3), round(model_c.conf_int().loc['F'][1], 3)]
    }

    # Monte Carlo simulation for indirect effect a × b
    np.random.seed(42)
    n_sim = 20000
    a_sim = np.random.normal(results['a (F → E)']['B'], results['a (F → E)']['SE'], n_sim)
    b_sim = np.random.normal(results['b (E → CW)']['B'], results['b (E → CW)']['SE'], n_sim)
    ab_sim = a_sim * b_sim
    ab_mean = np.mean(ab_sim)
    ci_lower = np.percentile(ab_sim, 2.5)
    ci_upper = np.percentile(ab_sim, 97.5)

    results['Indirect (a×b)'] = {
        'B': round(ab_mean, 4),
        '95% CI': f"[{round(ci_lower, 4)}, {round(ci_upper, 4)}]"
    }

    return results


# 运行两组分析
results_group0 = mlm_mediation(df_group0)
results_group1 = mlm_mediation(df_group1)

# 汇总输出
df_results = pd.DataFrame([results_group0, results_group1],
                          index=["Peer Friendship (Group 0)", "AI Friendship (Group 1)"])
print("\n📊 Group-wise Multilevel Mediation Results")
print(df_results)

# 可选保存
df_results.to_excel(r"D:\projects\AI & education\groupwise_mediation_resultsCW.xlsx")

# 读取数据
df = pd.read_excel(r"D:\projects\AI & education\data.xlsx")

# 数据预处理
df['ID'] = df['ID'].astype(str)
df['Time'] = df['Time'].astype(str)
df['Group'] = df['Group'].astype(str)
df['gender'] = df['gender'].astype(str)
df['hukou'] = df['hukou'].astype(str)

# 分组数据
df_group0 = df[df['Group'] == '0']  # Human group
df_group1 = df[df['Group'] == '1']  # AI group

# 定义函数：运行MLM并提取统计信息
def run_mlm_path_with_interaction(df_subset, y_variable):
    model = smf.mixedlm(f"{y_variable} ~ F * gender + Time + hukou",
                        df_subset,
                        groups=df_subset["ID"]).fit()

    # Friendship 主效应
    coef_f = model.params.get('F', None)
    se_f = model.bse.get('F', None)
    z_f = coef_f / se_f if coef_f is not None else None
    ci_f = f"[{round(coef_f - 1.96 * se_f, 3)}, {round(coef_f + 1.96 * se_f, 3)}]" if se_f else None

    # Friendship × Gender 交互效应
    coef_int = model.params.get('F:gender[T.2]', None)
    se_int = model.bse.get('F:gender[T.2]', None)
    z_int = coef_int / se_int if coef_int is not None else None
    p_int = model.pvalues.get('F:gender[T.2]', None)
    ci_int = f"[{round(coef_int - 1.96 * se_int, 3)}, {round(coef_int + 1.96 * se_int, 3)}]" if se_int else None

    return (round(coef_f, 3), round(se_f, 3), round(z_f, 3), ci_f,
            round(coef_int, 3), round(se_int, 3), round(z_int, 3), round(p_int, 4), ci_int)

# 运行分析并保存结果
results = []
for group, df_sub in [("Human (0)", df_group0), ("AI (1)", df_group1)]:
    for y in ["E", "I", "CW"]:
        (coef_f, se_f, z_f, ci_f,
         coef_int, se_int, z_int, p_int, ci_int) = run_mlm_path_with_interaction(df_sub, y)

        results.append({
            "Group": group,
            "Outcome": y,
            "B (F)": coef_f,
            "SE (F)": se_f,
            "z (F)": z_f,
            "95% CI (F)": ci_f,
            "Interaction (F × Gender)": coef_int,
            "SE (Int)": se_int,
            "z (Int)": z_int,
            "p (Int)": p_int,
            "95% CI (Int)": ci_int
        })

# 转换为 DataFrame 并输出
interaction_table = pd.DataFrame(results)

print("\n📊 分组MLM路径分析表（含 F × gender 交互项）")
print(interaction_table.to_string(index=False))

# 可选保存
interaction_table.to_excel(r"D:\projects\AI & education\groupwise_mlm_with_interaction.xlsx", index=False)

import pandas as pd
import statsmodels.formula.api as smf

# 读取数据
df = pd.read_excel(r"D:\projects\AI & education\data.xlsx")

# 数据预处理
df['ID'] = df['ID'].astype(str)
df['Time'] = df['Time'].astype(str)
df['Group'] = df['Group'].astype(str)
df['gender'] = df['gender'].astype(str)
df['hukou'] = df['hukou'].astype(str)

# 分组数据
df_group0 = df[df['Group'] == '0']  # Human group
df_group1 = df[df['Group'] == '1']  # AI group

# 定义函数：运行MLM并提取 F × hukou 的交互项统计信息
def run_mlm_path_with_hukou_interaction(df_subset, y_variable):
    model = smf.mixedlm(f"{y_variable} ~ F * hukou + Time + gender",
                        df_subset,
                        groups=df_subset["ID"]).fit()

    # Friendship 主效应
    coef_f = model.params.get('F', None)
    se_f = model.bse.get('F', None)
    z_f = coef_f / se_f if coef_f is not None else None
    ci_f = f"[{round(coef_f - 1.96 * se_f, 3)}, {round(coef_f + 1.96 * se_f, 3)}]" if se_f else None

    # Friendship × hukou 交互项
    coef_int = model.params.get('F:hukou[T.2]', None)
    se_int = model.bse.get('F:hukou[T.2]', None)
    z_int = coef_int / se_int if coef_int is not None else None
    p_int = model.pvalues.get('F:hukou[T.2]', None)
    ci_int = f"[{round(coef_int - 1.96 * se_int, 3)}, {round(coef_int + 1.96 * se_int, 3)}]" if se_int else None

    return (round(coef_f, 3), round(se_f, 3), round(z_f, 3), ci_f,
            round(coef_int, 3), round(se_int, 3), round(z_int, 3), round(p_int, 4), ci_int)

# 运行分析并保存结果
results = []
for group, df_sub in [("Human (0)", df_group0), ("AI (1)", df_group1)]:
    for y in ["E", "I", "CW"]:
        (coef_f, se_f, z_f, ci_f,
         coef_int, se_int, z_int, p_int, ci_int) = run_mlm_path_with_hukou_interaction(df_sub, y)

        results.append({
            "Group": group,
            "Outcome": y,
            "B (F)": coef_f,
            "SE (F)": se_f,
            "z (F)": z_f,
            "95% CI (F)": ci_f,
            "Interaction (F × hukou)": coef_int,
            "SE (Int)": se_int,
            "z (Int)": z_int,
            "p (Int)": p_int,
            "95% CI (Int)": ci_int
        })

# 输出表格
interaction_table = pd.DataFrame(results)

print("\n📊 分组MLM路径分析表（含 F × hukou 交互项）")
print(interaction_table.to_string(index=False))

# 可选保存
interaction_table.to_excel(r"D:\projects\AI & education\groupwise_mlm_FxHukou.xlsx", index=False)
