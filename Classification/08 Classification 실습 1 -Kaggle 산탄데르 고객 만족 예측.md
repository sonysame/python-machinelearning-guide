# 캐글 실습: 산탄데르 고객 만족 예측


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

cust_df=pd.read_csv(r'C:\Users\user\Data_Handling\santander-customer-satisfaction\train.csv')
print(cust_df.shape)
print(cust_df.info(verbose=True, null_counts=True))
cust_df.head(10)
```

    (76020, 371)
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 76020 entries, 0 to 76019
    Data columns (total 371 columns):
     #   Column                         Non-Null Count  Dtype  
    ---  ------                         --------------  -----  
     0   ID                             76020 non-null  int64  
     1   var3                           76020 non-null  int64  
     2   var15                          76020 non-null  int64  
     3   imp_ent_var16_ult1             76020 non-null  float64
     4   imp_op_var39_comer_ult1        76020 non-null  float64
     5   imp_op_var39_comer_ult3        76020 non-null  float64
     6   imp_op_var40_comer_ult1        76020 non-null  float64
     7   imp_op_var40_comer_ult3        76020 non-null  float64
     8   imp_op_var40_efect_ult1        76020 non-null  float64
     9   imp_op_var40_efect_ult3        76020 non-null  float64
     10  imp_op_var40_ult1              76020 non-null  float64
     11  imp_op_var41_comer_ult1        76020 non-null  float64
     12  imp_op_var41_comer_ult3        76020 non-null  float64
     13  imp_op_var41_efect_ult1        76020 non-null  float64
     14  imp_op_var41_efect_ult3        76020 non-null  float64
     15  imp_op_var41_ult1              76020 non-null  float64
     16  imp_op_var39_efect_ult1        76020 non-null  float64
     17  imp_op_var39_efect_ult3        76020 non-null  float64
     18  imp_op_var39_ult1              76020 non-null  float64
     19  imp_sal_var16_ult1             76020 non-null  float64
     20  ind_var1_0                     76020 non-null  int64  
     21  ind_var1                       76020 non-null  int64  
     22  ind_var2_0                     76020 non-null  int64  
     23  ind_var2                       76020 non-null  int64  
     24  ind_var5_0                     76020 non-null  int64  
     25  ind_var5                       76020 non-null  int64  
     26  ind_var6_0                     76020 non-null  int64  
     27  ind_var6                       76020 non-null  int64  
     28  ind_var8_0                     76020 non-null  int64  
     29  ind_var8                       76020 non-null  int64  
     30  ind_var12_0                    76020 non-null  int64  
     31  ind_var12                      76020 non-null  int64  
     32  ind_var13_0                    76020 non-null  int64  
     33  ind_var13_corto_0              76020 non-null  int64  
     34  ind_var13_corto                76020 non-null  int64  
     35  ind_var13_largo_0              76020 non-null  int64  
     36  ind_var13_largo                76020 non-null  int64  
     37  ind_var13_medio_0              76020 non-null  int64  
     38  ind_var13_medio                76020 non-null  int64  
     39  ind_var13                      76020 non-null  int64  
     40  ind_var14_0                    76020 non-null  int64  
     41  ind_var14                      76020 non-null  int64  
     42  ind_var17_0                    76020 non-null  int64  
     43  ind_var17                      76020 non-null  int64  
     44  ind_var18_0                    76020 non-null  int64  
     45  ind_var18                      76020 non-null  int64  
     46  ind_var19                      76020 non-null  int64  
     47  ind_var20_0                    76020 non-null  int64  
     48  ind_var20                      76020 non-null  int64  
     49  ind_var24_0                    76020 non-null  int64  
     50  ind_var24                      76020 non-null  int64  
     51  ind_var25_cte                  76020 non-null  int64  
     52  ind_var26_0                    76020 non-null  int64  
     53  ind_var26_cte                  76020 non-null  int64  
     54  ind_var26                      76020 non-null  int64  
     55  ind_var25_0                    76020 non-null  int64  
     56  ind_var25                      76020 non-null  int64  
     57  ind_var27_0                    76020 non-null  int64  
     58  ind_var28_0                    76020 non-null  int64  
     59  ind_var28                      76020 non-null  int64  
     60  ind_var27                      76020 non-null  int64  
     61  ind_var29_0                    76020 non-null  int64  
     62  ind_var29                      76020 non-null  int64  
     63  ind_var30_0                    76020 non-null  int64  
     64  ind_var30                      76020 non-null  int64  
     65  ind_var31_0                    76020 non-null  int64  
     66  ind_var31                      76020 non-null  int64  
     67  ind_var32_cte                  76020 non-null  int64  
     68  ind_var32_0                    76020 non-null  int64  
     69  ind_var32                      76020 non-null  int64  
     70  ind_var33_0                    76020 non-null  int64  
     71  ind_var33                      76020 non-null  int64  
     72  ind_var34_0                    76020 non-null  int64  
     73  ind_var34                      76020 non-null  int64  
     74  ind_var37_cte                  76020 non-null  int64  
     75  ind_var37_0                    76020 non-null  int64  
     76  ind_var37                      76020 non-null  int64  
     77  ind_var39_0                    76020 non-null  int64  
     78  ind_var40_0                    76020 non-null  int64  
     79  ind_var40                      76020 non-null  int64  
     80  ind_var41_0                    76020 non-null  int64  
     81  ind_var41                      76020 non-null  int64  
     82  ind_var39                      76020 non-null  int64  
     83  ind_var44_0                    76020 non-null  int64  
     84  ind_var44                      76020 non-null  int64  
     85  ind_var46_0                    76020 non-null  int64  
     86  ind_var46                      76020 non-null  int64  
     87  num_var1_0                     76020 non-null  int64  
     88  num_var1                       76020 non-null  int64  
     89  num_var4                       76020 non-null  int64  
     90  num_var5_0                     76020 non-null  int64  
     91  num_var5                       76020 non-null  int64  
     92  num_var6_0                     76020 non-null  int64  
     93  num_var6                       76020 non-null  int64  
     94  num_var8_0                     76020 non-null  int64  
     95  num_var8                       76020 non-null  int64  
     96  num_var12_0                    76020 non-null  int64  
     97  num_var12                      76020 non-null  int64  
     98  num_var13_0                    76020 non-null  int64  
     99  num_var13_corto_0              76020 non-null  int64  
     100 num_var13_corto                76020 non-null  int64  
     101 num_var13_largo_0              76020 non-null  int64  
     102 num_var13_largo                76020 non-null  int64  
     103 num_var13_medio_0              76020 non-null  int64  
     104 num_var13_medio                76020 non-null  int64  
     105 num_var13                      76020 non-null  int64  
     106 num_var14_0                    76020 non-null  int64  
     107 num_var14                      76020 non-null  int64  
     108 num_var17_0                    76020 non-null  int64  
     109 num_var17                      76020 non-null  int64  
     110 num_var18_0                    76020 non-null  int64  
     111 num_var18                      76020 non-null  int64  
     112 num_var20_0                    76020 non-null  int64  
     113 num_var20                      76020 non-null  int64  
     114 num_var24_0                    76020 non-null  int64  
     115 num_var24                      76020 non-null  int64  
     116 num_var26_0                    76020 non-null  int64  
     117 num_var26                      76020 non-null  int64  
     118 num_var25_0                    76020 non-null  int64  
     119 num_var25                      76020 non-null  int64  
     120 num_op_var40_hace2             76020 non-null  int64  
     121 num_op_var40_hace3             76020 non-null  int64  
     122 num_op_var40_ult1              76020 non-null  int64  
     123 num_op_var40_ult3              76020 non-null  int64  
     124 num_op_var41_hace2             76020 non-null  int64  
     125 num_op_var41_hace3             76020 non-null  int64  
     126 num_op_var41_ult1              76020 non-null  int64  
     127 num_op_var41_ult3              76020 non-null  int64  
     128 num_op_var39_hace2             76020 non-null  int64  
     129 num_op_var39_hace3             76020 non-null  int64  
     130 num_op_var39_ult1              76020 non-null  int64  
     131 num_op_var39_ult3              76020 non-null  int64  
     132 num_var27_0                    76020 non-null  int64  
     133 num_var28_0                    76020 non-null  int64  
     134 num_var28                      76020 non-null  int64  
     135 num_var27                      76020 non-null  int64  
     136 num_var29_0                    76020 non-null  int64  
     137 num_var29                      76020 non-null  int64  
     138 num_var30_0                    76020 non-null  int64  
     139 num_var30                      76020 non-null  int64  
     140 num_var31_0                    76020 non-null  int64  
     141 num_var31                      76020 non-null  int64  
     142 num_var32_0                    76020 non-null  int64  
     143 num_var32                      76020 non-null  int64  
     144 num_var33_0                    76020 non-null  int64  
     145 num_var33                      76020 non-null  int64  
     146 num_var34_0                    76020 non-null  int64  
     147 num_var34                      76020 non-null  int64  
     148 num_var35                      76020 non-null  int64  
     149 num_var37_med_ult2             76020 non-null  int64  
     150 num_var37_0                    76020 non-null  int64  
     151 num_var37                      76020 non-null  int64  
     152 num_var39_0                    76020 non-null  int64  
     153 num_var40_0                    76020 non-null  int64  
     154 num_var40                      76020 non-null  int64  
     155 num_var41_0                    76020 non-null  int64  
     156 num_var41                      76020 non-null  int64  
     157 num_var39                      76020 non-null  int64  
     158 num_var42_0                    76020 non-null  int64  
     159 num_var42                      76020 non-null  int64  
     160 num_var44_0                    76020 non-null  int64  
     161 num_var44                      76020 non-null  int64  
     162 num_var46_0                    76020 non-null  int64  
     163 num_var46                      76020 non-null  int64  
     164 saldo_var1                     76020 non-null  float64
     165 saldo_var5                     76020 non-null  float64
     166 saldo_var6                     76020 non-null  float64
     167 saldo_var8                     76020 non-null  float64
     168 saldo_var12                    76020 non-null  float64
     169 saldo_var13_corto              76020 non-null  float64
     170 saldo_var13_largo              76020 non-null  float64
     171 saldo_var13_medio              76020 non-null  int64  
     172 saldo_var13                    76020 non-null  float64
     173 saldo_var14                    76020 non-null  float64
     174 saldo_var17                    76020 non-null  float64
     175 saldo_var18                    76020 non-null  int64  
     176 saldo_var20                    76020 non-null  float64
     177 saldo_var24                    76020 non-null  float64
     178 saldo_var26                    76020 non-null  float64
     179 saldo_var25                    76020 non-null  float64
     180 saldo_var28                    76020 non-null  int64  
     181 saldo_var27                    76020 non-null  int64  
     182 saldo_var29                    76020 non-null  float64
     183 saldo_var30                    76020 non-null  float64
     184 saldo_var31                    76020 non-null  float64
     185 saldo_var32                    76020 non-null  float64
     186 saldo_var33                    76020 non-null  float64
     187 saldo_var34                    76020 non-null  int64  
     188 saldo_var37                    76020 non-null  float64
     189 saldo_var40                    76020 non-null  float64
     190 saldo_var41                    76020 non-null  int64  
     191 saldo_var42                    76020 non-null  float64
     192 saldo_var44                    76020 non-null  float64
     193 saldo_var46                    76020 non-null  int64  
     194 var36                          76020 non-null  int64  
     195 delta_imp_amort_var18_1y3      76020 non-null  int64  
     196 delta_imp_amort_var34_1y3      76020 non-null  int64  
     197 delta_imp_aport_var13_1y3      76020 non-null  float64
     198 delta_imp_aport_var17_1y3      76020 non-null  float64
     199 delta_imp_aport_var33_1y3      76020 non-null  float64
     200 delta_imp_compra_var44_1y3     76020 non-null  float64
     201 delta_imp_reemb_var13_1y3      76020 non-null  int64  
     202 delta_imp_reemb_var17_1y3      76020 non-null  int64  
     203 delta_imp_reemb_var33_1y3      76020 non-null  int64  
     204 delta_imp_trasp_var17_in_1y3   76020 non-null  int64  
     205 delta_imp_trasp_var17_out_1y3  76020 non-null  int64  
     206 delta_imp_trasp_var33_in_1y3   76020 non-null  int64  
     207 delta_imp_trasp_var33_out_1y3  76020 non-null  int64  
     208 delta_imp_venta_var44_1y3      76020 non-null  float64
     209 delta_num_aport_var13_1y3      76020 non-null  float64
     210 delta_num_aport_var17_1y3      76020 non-null  float64
     211 delta_num_aport_var33_1y3      76020 non-null  float64
     212 delta_num_compra_var44_1y3     76020 non-null  float64
     213 delta_num_reemb_var13_1y3      76020 non-null  int64  
     214 delta_num_reemb_var17_1y3      76020 non-null  int64  
     215 delta_num_reemb_var33_1y3      76020 non-null  int64  
     216 delta_num_trasp_var17_in_1y3   76020 non-null  int64  
     217 delta_num_trasp_var17_out_1y3  76020 non-null  int64  
     218 delta_num_trasp_var33_in_1y3   76020 non-null  int64  
     219 delta_num_trasp_var33_out_1y3  76020 non-null  int64  
     220 delta_num_venta_var44_1y3      76020 non-null  float64
     221 imp_amort_var18_hace3          76020 non-null  int64  
     222 imp_amort_var18_ult1           76020 non-null  float64
     223 imp_amort_var34_hace3          76020 non-null  int64  
     224 imp_amort_var34_ult1           76020 non-null  float64
     225 imp_aport_var13_hace3          76020 non-null  float64
     226 imp_aport_var13_ult1           76020 non-null  float64
     227 imp_aport_var17_hace3          76020 non-null  float64
     228 imp_aport_var17_ult1           76020 non-null  float64
     229 imp_aport_var33_hace3          76020 non-null  int64  
     230 imp_aport_var33_ult1           76020 non-null  int64  
     231 imp_var7_emit_ult1             76020 non-null  float64
     232 imp_var7_recib_ult1            76020 non-null  float64
     233 imp_compra_var44_hace3         76020 non-null  float64
     234 imp_compra_var44_ult1          76020 non-null  float64
     235 imp_reemb_var13_hace3          76020 non-null  int64  
     236 imp_reemb_var13_ult1           76020 non-null  float64
     237 imp_reemb_var17_hace3          76020 non-null  float64
     238 imp_reemb_var17_ult1           76020 non-null  float64
     239 imp_reemb_var33_hace3          76020 non-null  int64  
     240 imp_reemb_var33_ult1           76020 non-null  int64  
     241 imp_var43_emit_ult1            76020 non-null  float64
     242 imp_trans_var37_ult1           76020 non-null  float64
     243 imp_trasp_var17_in_hace3       76020 non-null  float64
     244 imp_trasp_var17_in_ult1        76020 non-null  float64
     245 imp_trasp_var17_out_hace3      76020 non-null  int64  
     246 imp_trasp_var17_out_ult1       76020 non-null  float64
     247 imp_trasp_var33_in_hace3       76020 non-null  float64
     248 imp_trasp_var33_in_ult1        76020 non-null  float64
     249 imp_trasp_var33_out_hace3      76020 non-null  int64  
     250 imp_trasp_var33_out_ult1       76020 non-null  int64  
     251 imp_venta_var44_hace3          76020 non-null  float64
     252 imp_venta_var44_ult1           76020 non-null  float64
     253 ind_var7_emit_ult1             76020 non-null  int64  
     254 ind_var7_recib_ult1            76020 non-null  int64  
     255 ind_var10_ult1                 76020 non-null  int64  
     256 ind_var10cte_ult1              76020 non-null  int64  
     257 ind_var9_cte_ult1              76020 non-null  int64  
     258 ind_var9_ult1                  76020 non-null  int64  
     259 ind_var43_emit_ult1            76020 non-null  int64  
     260 ind_var43_recib_ult1           76020 non-null  int64  
     261 var21                          76020 non-null  int64  
     262 num_var2_0_ult1                76020 non-null  int64  
     263 num_var2_ult1                  76020 non-null  int64  
     264 num_aport_var13_hace3          76020 non-null  int64  
     265 num_aport_var13_ult1           76020 non-null  int64  
     266 num_aport_var17_hace3          76020 non-null  int64  
     267 num_aport_var17_ult1           76020 non-null  int64  
     268 num_aport_var33_hace3          76020 non-null  int64  
     269 num_aport_var33_ult1           76020 non-null  int64  
     270 num_var7_emit_ult1             76020 non-null  int64  
     271 num_var7_recib_ult1            76020 non-null  int64  
     272 num_compra_var44_hace3         76020 non-null  int64  
     273 num_compra_var44_ult1          76020 non-null  int64  
     274 num_ent_var16_ult1             76020 non-null  int64  
     275 num_var22_hace2                76020 non-null  int64  
     276 num_var22_hace3                76020 non-null  int64  
     277 num_var22_ult1                 76020 non-null  int64  
     278 num_var22_ult3                 76020 non-null  int64  
     279 num_med_var22_ult3             76020 non-null  int64  
     280 num_med_var45_ult3             76020 non-null  int64  
     281 num_meses_var5_ult3            76020 non-null  int64  
     282 num_meses_var8_ult3            76020 non-null  int64  
     283 num_meses_var12_ult3           76020 non-null  int64  
     284 num_meses_var13_corto_ult3     76020 non-null  int64  
     285 num_meses_var13_largo_ult3     76020 non-null  int64  
     286 num_meses_var13_medio_ult3     76020 non-null  int64  
     287 num_meses_var17_ult3           76020 non-null  int64  
     288 num_meses_var29_ult3           76020 non-null  int64  
     289 num_meses_var33_ult3           76020 non-null  int64  
     290 num_meses_var39_vig_ult3       76020 non-null  int64  
     291 num_meses_var44_ult3           76020 non-null  int64  
     292 num_op_var39_comer_ult1        76020 non-null  int64  
     293 num_op_var39_comer_ult3        76020 non-null  int64  
     294 num_op_var40_comer_ult1        76020 non-null  int64  
     295 num_op_var40_comer_ult3        76020 non-null  int64  
     296 num_op_var40_efect_ult1        76020 non-null  int64  
     297 num_op_var40_efect_ult3        76020 non-null  int64  
     298 num_op_var41_comer_ult1        76020 non-null  int64  
     299 num_op_var41_comer_ult3        76020 non-null  int64  
     300 num_op_var41_efect_ult1        76020 non-null  int64  
     301 num_op_var41_efect_ult3        76020 non-null  int64  
     302 num_op_var39_efect_ult1        76020 non-null  int64  
     303 num_op_var39_efect_ult3        76020 non-null  int64  
     304 num_reemb_var13_hace3          76020 non-null  int64  
     305 num_reemb_var13_ult1           76020 non-null  int64  
     306 num_reemb_var17_hace3          76020 non-null  int64  
     307 num_reemb_var17_ult1           76020 non-null  int64  
     308 num_reemb_var33_hace3          76020 non-null  int64  
     309 num_reemb_var33_ult1           76020 non-null  int64  
     310 num_sal_var16_ult1             76020 non-null  int64  
     311 num_var43_emit_ult1            76020 non-null  int64  
     312 num_var43_recib_ult1           76020 non-null  int64  
     313 num_trasp_var11_ult1           76020 non-null  int64  
     314 num_trasp_var17_in_hace3       76020 non-null  int64  
     315 num_trasp_var17_in_ult1        76020 non-null  int64  
     316 num_trasp_var17_out_hace3      76020 non-null  int64  
     317 num_trasp_var17_out_ult1       76020 non-null  int64  
     318 num_trasp_var33_in_hace3       76020 non-null  int64  
     319 num_trasp_var33_in_ult1        76020 non-null  int64  
     320 num_trasp_var33_out_hace3      76020 non-null  int64  
     321 num_trasp_var33_out_ult1       76020 non-null  int64  
     322 num_venta_var44_hace3          76020 non-null  int64  
     323 num_venta_var44_ult1           76020 non-null  int64  
     324 num_var45_hace2                76020 non-null  int64  
     325 num_var45_hace3                76020 non-null  int64  
     326 num_var45_ult1                 76020 non-null  int64  
     327 num_var45_ult3                 76020 non-null  int64  
     328 saldo_var2_ult1                76020 non-null  int64  
     329 saldo_medio_var5_hace2         76020 non-null  float64
     330 saldo_medio_var5_hace3         76020 non-null  float64
     331 saldo_medio_var5_ult1          76020 non-null  float64
     332 saldo_medio_var5_ult3          76020 non-null  float64
     333 saldo_medio_var8_hace2         76020 non-null  float64
     334 saldo_medio_var8_hace3         76020 non-null  float64
     335 saldo_medio_var8_ult1          76020 non-null  float64
     336 saldo_medio_var8_ult3          76020 non-null  float64
     337 saldo_medio_var12_hace2        76020 non-null  float64
     338 saldo_medio_var12_hace3        76020 non-null  float64
     339 saldo_medio_var12_ult1         76020 non-null  float64
     340 saldo_medio_var12_ult3         76020 non-null  float64
     341 saldo_medio_var13_corto_hace2  76020 non-null  float64
     342 saldo_medio_var13_corto_hace3  76020 non-null  float64
     343 saldo_medio_var13_corto_ult1   76020 non-null  float64
     344 saldo_medio_var13_corto_ult3   76020 non-null  float64
     345 saldo_medio_var13_largo_hace2  76020 non-null  float64
     346 saldo_medio_var13_largo_hace3  76020 non-null  float64
     347 saldo_medio_var13_largo_ult1   76020 non-null  float64
     348 saldo_medio_var13_largo_ult3   76020 non-null  float64
     349 saldo_medio_var13_medio_hace2  76020 non-null  float64
     350 saldo_medio_var13_medio_hace3  76020 non-null  int64  
     351 saldo_medio_var13_medio_ult1   76020 non-null  int64  
     352 saldo_medio_var13_medio_ult3   76020 non-null  float64
     353 saldo_medio_var17_hace2        76020 non-null  float64
     354 saldo_medio_var17_hace3        76020 non-null  float64
     355 saldo_medio_var17_ult1         76020 non-null  float64
     356 saldo_medio_var17_ult3         76020 non-null  float64
     357 saldo_medio_var29_hace2        76020 non-null  float64
     358 saldo_medio_var29_hace3        76020 non-null  float64
     359 saldo_medio_var29_ult1         76020 non-null  float64
     360 saldo_medio_var29_ult3         76020 non-null  float64
     361 saldo_medio_var33_hace2        76020 non-null  float64
     362 saldo_medio_var33_hace3        76020 non-null  float64
     363 saldo_medio_var33_ult1         76020 non-null  float64
     364 saldo_medio_var33_ult3         76020 non-null  float64
     365 saldo_medio_var44_hace2        76020 non-null  float64
     366 saldo_medio_var44_hace3        76020 non-null  float64
     367 saldo_medio_var44_ult1         76020 non-null  float64
     368 saldo_medio_var44_ult3         76020 non-null  float64
     369 var38                          76020 non-null  float64
     370 TARGET                         76020 non-null  int64  
    dtypes: float64(111), int64(260)
    memory usage: 215.2 MB
    None
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>var3</th>
      <th>var15</th>
      <th>imp_ent_var16_ult1</th>
      <th>imp_op_var39_comer_ult1</th>
      <th>imp_op_var39_comer_ult3</th>
      <th>imp_op_var40_comer_ult1</th>
      <th>imp_op_var40_comer_ult3</th>
      <th>imp_op_var40_efect_ult1</th>
      <th>imp_op_var40_efect_ult3</th>
      <th>...</th>
      <th>saldo_medio_var33_hace2</th>
      <th>saldo_medio_var33_hace3</th>
      <th>saldo_medio_var33_ult1</th>
      <th>saldo_medio_var33_ult3</th>
      <th>saldo_medio_var44_hace2</th>
      <th>saldo_medio_var44_hace3</th>
      <th>saldo_medio_var44_ult1</th>
      <th>saldo_medio_var44_ult3</th>
      <th>var38</th>
      <th>TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
      <td>23</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>39205.170000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>2</td>
      <td>34</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>49278.030000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>2</td>
      <td>23</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>67333.770000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8</td>
      <td>2</td>
      <td>37</td>
      <td>0.0</td>
      <td>195.0</td>
      <td>195.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>64007.970000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10</td>
      <td>2</td>
      <td>39</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>117310.979016</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>13</td>
      <td>2</td>
      <td>23</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>87975.750000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>14</td>
      <td>2</td>
      <td>27</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>94956.660000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>18</td>
      <td>2</td>
      <td>26</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>251638.950000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>20</td>
      <td>2</td>
      <td>45</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>101962.020000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>23</td>
      <td>2</td>
      <td>25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>356463.060000</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 371 columns</p>
</div>




```python
print(cust_df['TARGET'].value_counts())
unsatisfied_cnt=cust_df[cust_df['TARGET']==1].TARGET.count()
total_cnt=cust_df.TARGET.count()
print("unsatisfied 비율: {0:.2f}".format((unsatisfied_cnt/total_cnt)))
```

    0    73012
    1     3008
    Name: TARGET, dtype: int64
    unsatisfied 비율: 0.04
    


```python
cust_df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>var3</th>
      <th>var15</th>
      <th>imp_ent_var16_ult1</th>
      <th>imp_op_var39_comer_ult1</th>
      <th>imp_op_var39_comer_ult3</th>
      <th>imp_op_var40_comer_ult1</th>
      <th>imp_op_var40_comer_ult3</th>
      <th>imp_op_var40_efect_ult1</th>
      <th>imp_op_var40_efect_ult3</th>
      <th>...</th>
      <th>saldo_medio_var33_hace2</th>
      <th>saldo_medio_var33_hace3</th>
      <th>saldo_medio_var33_ult1</th>
      <th>saldo_medio_var33_ult3</th>
      <th>saldo_medio_var44_hace2</th>
      <th>saldo_medio_var44_hace3</th>
      <th>saldo_medio_var44_ult1</th>
      <th>saldo_medio_var44_ult3</th>
      <th>var38</th>
      <th>TARGET</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>...</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>76020.000000</td>
      <td>7.602000e+04</td>
      <td>76020.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>75964.050723</td>
      <td>-1523.199277</td>
      <td>33.212865</td>
      <td>86.208265</td>
      <td>72.363067</td>
      <td>119.529632</td>
      <td>3.559130</td>
      <td>6.472698</td>
      <td>0.412946</td>
      <td>0.567352</td>
      <td>...</td>
      <td>7.935824</td>
      <td>1.365146</td>
      <td>12.215580</td>
      <td>8.784074</td>
      <td>31.505324</td>
      <td>1.858575</td>
      <td>76.026165</td>
      <td>56.614351</td>
      <td>1.172358e+05</td>
      <td>0.039569</td>
    </tr>
    <tr>
      <th>std</th>
      <td>43781.947379</td>
      <td>39033.462364</td>
      <td>12.956486</td>
      <td>1614.757313</td>
      <td>339.315831</td>
      <td>546.266294</td>
      <td>93.155749</td>
      <td>153.737066</td>
      <td>30.604864</td>
      <td>36.513513</td>
      <td>...</td>
      <td>455.887218</td>
      <td>113.959637</td>
      <td>783.207399</td>
      <td>538.439211</td>
      <td>2013.125393</td>
      <td>147.786584</td>
      <td>4040.337842</td>
      <td>2852.579397</td>
      <td>1.826646e+05</td>
      <td>0.194945</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>-999999.000000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.163750e+03</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>38104.750000</td>
      <td>2.000000</td>
      <td>23.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.787061e+04</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>76043.000000</td>
      <td>2.000000</td>
      <td>28.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.064092e+05</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>113748.750000</td>
      <td>2.000000</td>
      <td>40.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.187563e+05</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>151838.000000</td>
      <td>238.000000</td>
      <td>105.000000</td>
      <td>210000.000000</td>
      <td>12888.030000</td>
      <td>21024.810000</td>
      <td>8237.820000</td>
      <td>11073.570000</td>
      <td>6600.000000</td>
      <td>6600.000000</td>
      <td>...</td>
      <td>50003.880000</td>
      <td>20385.720000</td>
      <td>138831.630000</td>
      <td>91778.730000</td>
      <td>438329.220000</td>
      <td>24650.010000</td>
      <td>681462.900000</td>
      <td>397884.300000</td>
      <td>2.203474e+07</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 371 columns</p>
</div>



var3 칼럼의 경우 min값이 -999999다. 이는 NaN이나 특정 예외값을 -999999로 변환했을 것이다.


```python
cust_df['var3'].value_counts()
```




     2         74165
     8           138
    -999999      116
     9           110
     3           108
               ...  
     177           1
     87            1
     151           1
     215           1
     191           1
    Name: var3, Length: 208, dtype: int64



### 데이터 전처리
* -999999인 값이 116개가 있는데, 이는 다른 값에 비해 너무 편차가 심하므로 -999999을 가장 값이 많은 2로 변환
* ID 피처는 단순 식별자에 불과하므로 피처 드롭


```python
cust_df['var3'].replace(-999999, 2, inplace=True)
cust_df.drop('ID',axis=1, inplace=True) #피처 드롭시 axis=1 설정!!
```


```python
X_features=cust_df.iloc[:,:-1] #마지막 TARGET 컬럼 제외
y_labels=cust_df.iloc[:,-1]    #마지막 TARGET 컬럼
print("피처 데이터 shape:{0}".format(X_features.shape))
```

    피처 데이터 shape:(76020, 369)
    

학습과 테스트 데이터 세트 모두 TARGET의 값의 분포가 원본 데이터와 유사하게 전체 데이터의 4% 정도의 불만족 값으로 만들어졌음


```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(X_features, y_labels, test_size=0.2, random_state=0)

train_cnt=y_train.count()
test_cnt=y_test.count()
print("학습 세트 Shape:{0}, 테스트 세트 Shape:{1}".format(X_train.shape, X_test.shape))
print("학습 세트 레이블 값 분포 비율")
print(y_train.value_counts()/train_cnt)
print("\n테이블 세트 레이블 값 분포 비율")
print(y_test.value_counts()/test_cnt)
```

    학습 세트 Shape:(60816, 369), 테스트 세트 Shape:(15204, 369)
    학습 세트 레이블 값 분포 비율
    0    0.960964
    1    0.039036
    Name: TARGET, dtype: float64
    
    테이블 세트 레이블 값 분포 비율
    0    0.9583
    1    0.0417
    Name: TARGET, dtype: float64
    

## XGBoost 모델 학습과 하이퍼 파라미터 튜닝


```python
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

xgb_clf=XGBClassifier(n_estimators=500, random_state=156)
xgb_clf.fit(X_train, y_train, early_stopping_rounds=100, eval_metric='auc', eval_set=[(X_train, y_train),(X_test, y_test)])
xgb_roc_score=roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:,1], average=None)
print("ROC AUC: {0:.4f}".format(xgb_roc_score))
```

    C:\Users\user\anaconda3\lib\site-packages\xgboost\sklearn.py:888: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)
    

    [0]	validation_0-auc:0.82005	validation_1-auc:0.81157
    [1]	validation_0-auc:0.83400	validation_1-auc:0.82452
    [2]	validation_0-auc:0.83870	validation_1-auc:0.82746
    [3]	validation_0-auc:0.84419	validation_1-auc:0.82922
    [4]	validation_0-auc:0.84783	validation_1-auc:0.83298
    [5]	validation_0-auc:0.85125	validation_1-auc:0.83500
    [6]	validation_0-auc:0.85501	validation_1-auc:0.83653
    [7]	validation_0-auc:0.85830	validation_1-auc:0.83782
    [8]	validation_0-auc:0.86143	validation_1-auc:0.83802
    [9]	validation_0-auc:0.86452	validation_1-auc:0.83914
    [10]	validation_0-auc:0.86717	validation_1-auc:0.83954
    [11]	validation_0-auc:0.87013	validation_1-auc:0.83983
    [12]	validation_0-auc:0.87369	validation_1-auc:0.84033
    [13]	validation_0-auc:0.87620	validation_1-auc:0.84054
    [14]	validation_0-auc:0.87799	validation_1-auc:0.84135
    [15]	validation_0-auc:0.88072	validation_1-auc:0.84117
    [16]	validation_0-auc:0.88237	validation_1-auc:0.84101
    [17]	validation_0-auc:0.88352	validation_1-auc:0.84071
    [18]	validation_0-auc:0.88457	validation_1-auc:0.84052
    [19]	validation_0-auc:0.88592	validation_1-auc:0.84023
    [20]	validation_0-auc:0.88788	validation_1-auc:0.84012
    [21]	validation_0-auc:0.88845	validation_1-auc:0.84022
    [22]	validation_0-auc:0.88980	validation_1-auc:0.84007
    [23]	validation_0-auc:0.89019	validation_1-auc:0.84009
    [24]	validation_0-auc:0.89193	validation_1-auc:0.83974
    [25]	validation_0-auc:0.89253	validation_1-auc:0.84015
    [26]	validation_0-auc:0.89329	validation_1-auc:0.84101
    [27]	validation_0-auc:0.89386	validation_1-auc:0.84088
    [28]	validation_0-auc:0.89416	validation_1-auc:0.84074
    [29]	validation_0-auc:0.89660	validation_1-auc:0.83999
    [30]	validation_0-auc:0.89738	validation_1-auc:0.83959
    [31]	validation_0-auc:0.89911	validation_1-auc:0.83952
    [32]	validation_0-auc:0.90103	validation_1-auc:0.83901
    [33]	validation_0-auc:0.90250	validation_1-auc:0.83885
    [34]	validation_0-auc:0.90275	validation_1-auc:0.83887
    [35]	validation_0-auc:0.90290	validation_1-auc:0.83864
    [36]	validation_0-auc:0.90460	validation_1-auc:0.83834
    [37]	validation_0-auc:0.90497	validation_1-auc:0.83810
    [38]	validation_0-auc:0.90515	validation_1-auc:0.83810
    [39]	validation_0-auc:0.90533	validation_1-auc:0.83813
    [40]	validation_0-auc:0.90574	validation_1-auc:0.83776
    [41]	validation_0-auc:0.90690	validation_1-auc:0.83720
    [42]	validation_0-auc:0.90715	validation_1-auc:0.83684
    [43]	validation_0-auc:0.90736	validation_1-auc:0.83672
    [44]	validation_0-auc:0.90758	validation_1-auc:0.83674
    [45]	validation_0-auc:0.90767	validation_1-auc:0.83693
    [46]	validation_0-auc:0.90777	validation_1-auc:0.83686
    [47]	validation_0-auc:0.90791	validation_1-auc:0.83678
    [48]	validation_0-auc:0.90829	validation_1-auc:0.83694
    [49]	validation_0-auc:0.90869	validation_1-auc:0.83676
    [50]	validation_0-auc:0.90890	validation_1-auc:0.83655
    [51]	validation_0-auc:0.91067	validation_1-auc:0.83669
    [52]	validation_0-auc:0.91238	validation_1-auc:0.83641
    [53]	validation_0-auc:0.91352	validation_1-auc:0.83690
    [54]	validation_0-auc:0.91386	validation_1-auc:0.83693
    [55]	validation_0-auc:0.91406	validation_1-auc:0.83681
    [56]	validation_0-auc:0.91545	validation_1-auc:0.83680
    [57]	validation_0-auc:0.91556	validation_1-auc:0.83667
    [58]	validation_0-auc:0.91628	validation_1-auc:0.83664
    [59]	validation_0-auc:0.91725	validation_1-auc:0.83591
    [60]	validation_0-auc:0.91762	validation_1-auc:0.83576
    [61]	validation_0-auc:0.91784	validation_1-auc:0.83534
    [62]	validation_0-auc:0.91872	validation_1-auc:0.83513
    [63]	validation_0-auc:0.91892	validation_1-auc:0.83510
    [64]	validation_0-auc:0.91896	validation_1-auc:0.83508
    [65]	validation_0-auc:0.91907	validation_1-auc:0.83518
    [66]	validation_0-auc:0.91970	validation_1-auc:0.83510
    [67]	validation_0-auc:0.91982	validation_1-auc:0.83523
    [68]	validation_0-auc:0.92007	validation_1-auc:0.83457
    [69]	validation_0-auc:0.92015	validation_1-auc:0.83460
    [70]	validation_0-auc:0.92024	validation_1-auc:0.83446
    [71]	validation_0-auc:0.92037	validation_1-auc:0.83462
    [72]	validation_0-auc:0.92087	validation_1-auc:0.83394
    [73]	validation_0-auc:0.92094	validation_1-auc:0.83410
    [74]	validation_0-auc:0.92133	validation_1-auc:0.83394
    [75]	validation_0-auc:0.92141	validation_1-auc:0.83368
    [76]	validation_0-auc:0.92321	validation_1-auc:0.83413
    [77]	validation_0-auc:0.92415	validation_1-auc:0.83359
    [78]	validation_0-auc:0.92503	validation_1-auc:0.83353
    [79]	validation_0-auc:0.92539	validation_1-auc:0.83293
    [80]	validation_0-auc:0.92577	validation_1-auc:0.83253
    [81]	validation_0-auc:0.92677	validation_1-auc:0.83187
    [82]	validation_0-auc:0.92706	validation_1-auc:0.83230
    [83]	validation_0-auc:0.92800	validation_1-auc:0.83216
    [84]	validation_0-auc:0.92822	validation_1-auc:0.83206
    [85]	validation_0-auc:0.92870	validation_1-auc:0.83196
    [86]	validation_0-auc:0.92875	validation_1-auc:0.83200
    [87]	validation_0-auc:0.92881	validation_1-auc:0.83208
    [88]	validation_0-auc:0.92919	validation_1-auc:0.83174
    [89]	validation_0-auc:0.92940	validation_1-auc:0.83160
    [90]	validation_0-auc:0.92948	validation_1-auc:0.83155
    [91]	validation_0-auc:0.92959	validation_1-auc:0.83165
    [92]	validation_0-auc:0.92964	validation_1-auc:0.83172
    [93]	validation_0-auc:0.93031	validation_1-auc:0.83160
    [94]	validation_0-auc:0.93032	validation_1-auc:0.83150
    [95]	validation_0-auc:0.93037	validation_1-auc:0.83132
    [96]	validation_0-auc:0.93083	validation_1-auc:0.83090
    [97]	validation_0-auc:0.93091	validation_1-auc:0.83091
    [98]	validation_0-auc:0.93168	validation_1-auc:0.83066
    [99]	validation_0-auc:0.93245	validation_1-auc:0.83058
    [100]	validation_0-auc:0.93286	validation_1-auc:0.83029
    [101]	validation_0-auc:0.93361	validation_1-auc:0.82955
    [102]	validation_0-auc:0.93359	validation_1-auc:0.82962
    [103]	validation_0-auc:0.93435	validation_1-auc:0.82893
    [104]	validation_0-auc:0.93446	validation_1-auc:0.82837
    [105]	validation_0-auc:0.93480	validation_1-auc:0.82815
    [106]	validation_0-auc:0.93579	validation_1-auc:0.82744
    [107]	validation_0-auc:0.93583	validation_1-auc:0.82728
    [108]	validation_0-auc:0.93610	validation_1-auc:0.82651
    [109]	validation_0-auc:0.93617	validation_1-auc:0.82650
    [110]	validation_0-auc:0.93659	validation_1-auc:0.82621
    [111]	validation_0-auc:0.93663	validation_1-auc:0.82620
    [112]	validation_0-auc:0.93710	validation_1-auc:0.82591
    [113]	validation_0-auc:0.93781	validation_1-auc:0.82498
    [114]	validation_0-auc:0.93793	validation_1-auc:0.82525
    ROC AUC: 0.8413
    

average가 macro/weighted 이부분은 https://rython.tistory.com/14 참조!


```python
from sklearn.model_selection import GridSearchCV

xgb_clf=XGBClassifier(n_estimators=100)
params={'max_depth':[5,7],
        'min_child_weight':[1,3],
        'colsample_bytree':[0.5,0.75]}

gridcv=GridSearchCV(xgb_clf, param_grid=params, cv=3)
gridcv.fit(X_train, y_train, early_stopping_rounds=30, eval_metric="auc", eval_set=[(X_train, y_train),(X_test, y_test)])

print("GridSearchCV 최적 파라미머:",gridcv.best_params_)

xgb_roc_score=roc_auc_score(y_test, gridcv.predict_proba(X_test)[:,1], average='macro')

print("ROC AUC: {0:.4f}".format(xgb_roc_score))
```

    C:\Users\user\anaconda3\lib\site-packages\xgboost\sklearn.py:888: UserWarning: The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
      warnings.warn(label_encoder_deprecation_msg, UserWarning)
    

    [0]	validation_0-auc:0.79161	validation_1-auc:0.79321
    [1]	validation_0-auc:0.81865	validation_1-auc:0.81375
    [2]	validation_0-auc:0.82586	validation_1-auc:0.81846
    [3]	validation_0-auc:0.82789	validation_1-auc:0.82226
    [4]	validation_0-auc:0.83249	validation_1-auc:0.82677
    [5]	validation_0-auc:0.83477	validation_1-auc:0.83225
    [6]	validation_0-auc:0.83340	validation_1-auc:0.82654
    [7]	validation_0-auc:0.84223	validation_1-auc:0.83486
    [8]	validation_0-auc:0.84586	validation_1-auc:0.83682
    [9]	validation_0-auc:0.84557	validation_1-auc:0.83472
    [10]	validation_0-auc:0.84423	validation_1-auc:0.83181
    [11]	validation_0-auc:0.84428	validation_1-auc:0.82920
    [12]	validation_0-auc:0.85176	validation_1-auc:0.83433
    [13]	validation_0-auc:0.85540	validation_1-auc:0.83565
    [14]	validation_0-auc:0.85719	validation_1-auc:0.83696
    [15]	validation_0-auc:0.85849	validation_1-auc:0.83561
    [16]	validation_0-auc:0.85964	validation_1-auc:0.83578
    [17]	validation_0-auc:0.86092	validation_1-auc:0.83570
    [18]	validation_0-auc:0.86187	validation_1-auc:0.83595
    [19]	validation_0-auc:0.86251	validation_1-auc:0.83552
    [20]	validation_0-auc:0.86300	validation_1-auc:0.83452
    [21]	validation_0-auc:0.86376	validation_1-auc:0.83437
    [22]	validation_0-auc:0.86441	validation_1-auc:0.83516
    [23]	validation_0-auc:0.86552	validation_1-auc:0.83470
    [24]	validation_0-auc:0.86602	validation_1-auc:0.83492
    [25]	validation_0-auc:0.86698	validation_1-auc:0.83510
    [26]	validation_0-auc:0.86768	validation_1-auc:0.83412
    [27]	validation_0-auc:0.86855	validation_1-auc:0.83394
    [28]	validation_0-auc:0.86898	validation_1-auc:0.83441
    [29]	validation_0-auc:0.86914	validation_1-auc:0.83440
    [30]	validation_0-auc:0.86952	validation_1-auc:0.83380
    [31]	validation_0-auc:0.87050	validation_1-auc:0.83346
    [32]	validation_0-auc:0.87084	validation_1-auc:0.83334
    [33]	validation_0-auc:0.87112	validation_1-auc:0.83313
    [34]	validation_0-auc:0.87158	validation_1-auc:0.83383
    [35]	validation_0-auc:0.87172	validation_1-auc:0.83376
    [36]	validation_0-auc:0.87259	validation_1-auc:0.83340
    [37]	validation_0-auc:0.87307	validation_1-auc:0.83344
    [38]	validation_0-auc:0.87318	validation_1-auc:0.83343
    [39]	validation_0-auc:0.87337	validation_1-auc:0.83370
    [40]	validation_0-auc:0.87347	validation_1-auc:0.83373
    [41]	validation_0-auc:0.87408	validation_1-auc:0.83358
    [42]	validation_0-auc:0.87432	validation_1-auc:0.83325
    [43]	validation_0-auc:0.87431	validation_1-auc:0.83319
    [44]	validation_0-auc:0.87509	validation_1-auc:0.83344
    [0]	validation_0-auc:0.80013	validation_1-auc:0.79685
    [1]	validation_0-auc:0.82084	validation_1-auc:0.81574
    [2]	validation_0-auc:0.82744	validation_1-auc:0.82189
    [3]	validation_0-auc:0.83029	validation_1-auc:0.82317
    [4]	validation_0-auc:0.83578	validation_1-auc:0.82564
    [5]	validation_0-auc:0.83777	validation_1-auc:0.83385
    [6]	validation_0-auc:0.83742	validation_1-auc:0.83162
    [7]	validation_0-auc:0.84373	validation_1-auc:0.83436
    [8]	validation_0-auc:0.84835	validation_1-auc:0.83664
    [9]	validation_0-auc:0.84790	validation_1-auc:0.83583
    [10]	validation_0-auc:0.84717	validation_1-auc:0.83268
    [11]	validation_0-auc:0.84654	validation_1-auc:0.83066
    [12]	validation_0-auc:0.85377	validation_1-auc:0.83579
    [13]	validation_0-auc:0.85799	validation_1-auc:0.83859
    [14]	validation_0-auc:0.85962	validation_1-auc:0.83984
    [15]	validation_0-auc:0.86143	validation_1-auc:0.84003
    [16]	validation_0-auc:0.86269	validation_1-auc:0.84049
    [17]	validation_0-auc:0.86401	validation_1-auc:0.84009
    [18]	validation_0-auc:0.86474	validation_1-auc:0.84034
    [19]	validation_0-auc:0.86659	validation_1-auc:0.84138
    [20]	validation_0-auc:0.86728	validation_1-auc:0.84100
    [21]	validation_0-auc:0.86821	validation_1-auc:0.84058
    [22]	validation_0-auc:0.86943	validation_1-auc:0.84128
    [23]	validation_0-auc:0.86992	validation_1-auc:0.84122
    [24]	validation_0-auc:0.87036	validation_1-auc:0.84116
    [25]	validation_0-auc:0.87088	validation_1-auc:0.84045
    [26]	validation_0-auc:0.87134	validation_1-auc:0.83974
    [27]	validation_0-auc:0.87294	validation_1-auc:0.83926
    [28]	validation_0-auc:0.87305	validation_1-auc:0.83943
    [29]	validation_0-auc:0.87333	validation_1-auc:0.84017
    [30]	validation_0-auc:0.87443	validation_1-auc:0.83949
    [31]	validation_0-auc:0.87465	validation_1-auc:0.83936
    [32]	validation_0-auc:0.87512	validation_1-auc:0.83943
    [33]	validation_0-auc:0.87517	validation_1-auc:0.83951
    [34]	validation_0-auc:0.87541	validation_1-auc:0.83953
    [35]	validation_0-auc:0.87551	validation_1-auc:0.83946
    [36]	validation_0-auc:0.87581	validation_1-auc:0.83936
    [37]	validation_0-auc:0.87601	validation_1-auc:0.83919
    [38]	validation_0-auc:0.87620	validation_1-auc:0.83874
    [39]	validation_0-auc:0.87670	validation_1-auc:0.83844
    [40]	validation_0-auc:0.87679	validation_1-auc:0.83859
    [41]	validation_0-auc:0.87709	validation_1-auc:0.83830
    [42]	validation_0-auc:0.87736	validation_1-auc:0.83823
    [43]	validation_0-auc:0.87751	validation_1-auc:0.83796
    [44]	validation_0-auc:0.87774	validation_1-auc:0.83765
    [45]	validation_0-auc:0.87783	validation_1-auc:0.83786
    [46]	validation_0-auc:0.87799	validation_1-auc:0.83761
    [47]	validation_0-auc:0.87837	validation_1-auc:0.83698
    [48]	validation_0-auc:0.87864	validation_1-auc:0.83699
    [0]	validation_0-auc:0.80039	validation_1-auc:0.80013
    [1]	validation_0-auc:0.82111	validation_1-auc:0.82026
    [2]	validation_0-auc:0.82749	validation_1-auc:0.82627
    [3]	validation_0-auc:0.83124	validation_1-auc:0.82830
    [4]	validation_0-auc:0.83475	validation_1-auc:0.82881
    [5]	validation_0-auc:0.83676	validation_1-auc:0.83385
    [6]	validation_0-auc:0.83648	validation_1-auc:0.83085
    [7]	validation_0-auc:0.84336	validation_1-auc:0.83472
    [8]	validation_0-auc:0.84624	validation_1-auc:0.83404
    [9]	validation_0-auc:0.84541	validation_1-auc:0.83287
    [10]	validation_0-auc:0.84554	validation_1-auc:0.83039
    [11]	validation_0-auc:0.84525	validation_1-auc:0.82995
    [12]	validation_0-auc:0.85144	validation_1-auc:0.83489
    [13]	validation_0-auc:0.85525	validation_1-auc:0.83803
    [14]	validation_0-auc:0.85746	validation_1-auc:0.84145
    [15]	validation_0-auc:0.85818	validation_1-auc:0.84082
    [16]	validation_0-auc:0.86004	validation_1-auc:0.84076
    [17]	validation_0-auc:0.86126	validation_1-auc:0.84139
    [18]	validation_0-auc:0.86194	validation_1-auc:0.84041
    [19]	validation_0-auc:0.86338	validation_1-auc:0.84100
    [20]	validation_0-auc:0.86386	validation_1-auc:0.84145
    [21]	validation_0-auc:0.86552	validation_1-auc:0.84030
    [22]	validation_0-auc:0.86691	validation_1-auc:0.84072
    [23]	validation_0-auc:0.86766	validation_1-auc:0.84077
    [24]	validation_0-auc:0.86826	validation_1-auc:0.84136
    [25]	validation_0-auc:0.86940	validation_1-auc:0.84120
    [26]	validation_0-auc:0.87045	validation_1-auc:0.84098
    [27]	validation_0-auc:0.87065	validation_1-auc:0.84148
    [28]	validation_0-auc:0.87075	validation_1-auc:0.84120
    [29]	validation_0-auc:0.87115	validation_1-auc:0.84147
    [30]	validation_0-auc:0.87119	validation_1-auc:0.84181
    [31]	validation_0-auc:0.87147	validation_1-auc:0.84172
    [32]	validation_0-auc:0.87226	validation_1-auc:0.84100
    [33]	validation_0-auc:0.87241	validation_1-auc:0.84149
    [34]	validation_0-auc:0.87256	validation_1-auc:0.84120
    [35]	validation_0-auc:0.87297	validation_1-auc:0.84095
    [36]	validation_0-auc:0.87350	validation_1-auc:0.84051
    [37]	validation_0-auc:0.87395	validation_1-auc:0.84084
    [38]	validation_0-auc:0.87435	validation_1-auc:0.84055
    [39]	validation_0-auc:0.87450	validation_1-auc:0.84048
    [40]	validation_0-auc:0.87465	validation_1-auc:0.84042
    [41]	validation_0-auc:0.87487	validation_1-auc:0.84034
    [42]	validation_0-auc:0.87519	validation_1-auc:0.84021
    [43]	validation_0-auc:0.87525	validation_1-auc:0.84022
    [44]	validation_0-auc:0.87595	validation_1-auc:0.83967
    [45]	validation_0-auc:0.87630	validation_1-auc:0.84004
    [46]	validation_0-auc:0.87700	validation_1-auc:0.83966
    [47]	validation_0-auc:0.87743	validation_1-auc:0.83963
    [48]	validation_0-auc:0.87770	validation_1-auc:0.83931
    [49]	validation_0-auc:0.87782	validation_1-auc:0.83925
    [50]	validation_0-auc:0.87827	validation_1-auc:0.83935
    [51]	validation_0-auc:0.87862	validation_1-auc:0.83920
    [52]	validation_0-auc:0.87951	validation_1-auc:0.83895
    [53]	validation_0-auc:0.88027	validation_1-auc:0.83876
    [54]	validation_0-auc:0.88118	validation_1-auc:0.83840
    [55]	validation_0-auc:0.88128	validation_1-auc:0.83834
    [56]	validation_0-auc:0.88146	validation_1-auc:0.83873
    [57]	validation_0-auc:0.88158	validation_1-auc:0.83860
    [58]	validation_0-auc:0.88183	validation_1-auc:0.83810
    [59]	validation_0-auc:0.88192	validation_1-auc:0.83774
    [60]	validation_0-auc:0.88215	validation_1-auc:0.83723
    [0]	validation_0-auc:0.79210	validation_1-auc:0.79292
    [1]	validation_0-auc:0.81759	validation_1-auc:0.81404
    [2]	validation_0-auc:0.82567	validation_1-auc:0.81864
    [3]	validation_0-auc:0.82819	validation_1-auc:0.82244
    [4]	validation_0-auc:0.83233	validation_1-auc:0.82618
    [5]	validation_0-auc:0.83480	validation_1-auc:0.83163
    [6]	validation_0-auc:0.83342	validation_1-auc:0.82840
    [7]	validation_0-auc:0.84265	validation_1-auc:0.83512
    [8]	validation_0-auc:0.84614	validation_1-auc:0.83742
    [9]	validation_0-auc:0.84573	validation_1-auc:0.83475
    [10]	validation_0-auc:0.84426	validation_1-auc:0.83066
    [11]	validation_0-auc:0.84358	validation_1-auc:0.82937
    [12]	validation_0-auc:0.85089	validation_1-auc:0.83491
    [13]	validation_0-auc:0.85457	validation_1-auc:0.83785
    [14]	validation_0-auc:0.85644	validation_1-auc:0.83894
    [15]	validation_0-auc:0.85744	validation_1-auc:0.83784
    [16]	validation_0-auc:0.85870	validation_1-auc:0.83899
    [17]	validation_0-auc:0.86002	validation_1-auc:0.83854
    [18]	validation_0-auc:0.86092	validation_1-auc:0.83860
    [19]	validation_0-auc:0.86154	validation_1-auc:0.83818
    [20]	validation_0-auc:0.86190	validation_1-auc:0.83772
    [21]	validation_0-auc:0.86295	validation_1-auc:0.83703
    [22]	validation_0-auc:0.86334	validation_1-auc:0.83721
    [23]	validation_0-auc:0.86400	validation_1-auc:0.83581
    [24]	validation_0-auc:0.86454	validation_1-auc:0.83557
    [25]	validation_0-auc:0.86494	validation_1-auc:0.83534
    [26]	validation_0-auc:0.86514	validation_1-auc:0.83481
    [27]	validation_0-auc:0.86660	validation_1-auc:0.83557
    [28]	validation_0-auc:0.86784	validation_1-auc:0.83546
    [29]	validation_0-auc:0.86790	validation_1-auc:0.83545
    [30]	validation_0-auc:0.86838	validation_1-auc:0.83496
    [31]	validation_0-auc:0.86864	validation_1-auc:0.83481
    [32]	validation_0-auc:0.86882	validation_1-auc:0.83472
    [33]	validation_0-auc:0.86897	validation_1-auc:0.83482
    [34]	validation_0-auc:0.86908	validation_1-auc:0.83423
    [35]	validation_0-auc:0.86978	validation_1-auc:0.83350
    [36]	validation_0-auc:0.86993	validation_1-auc:0.83334
    [37]	validation_0-auc:0.86999	validation_1-auc:0.83365
    [38]	validation_0-auc:0.87021	validation_1-auc:0.83384
    [39]	validation_0-auc:0.87076	validation_1-auc:0.83373
    [40]	validation_0-auc:0.87093	validation_1-auc:0.83373
    [41]	validation_0-auc:0.87104	validation_1-auc:0.83359
    [42]	validation_0-auc:0.87173	validation_1-auc:0.83365
    [43]	validation_0-auc:0.87265	validation_1-auc:0.83386
    [44]	validation_0-auc:0.87338	validation_1-auc:0.83319
    [45]	validation_0-auc:0.87363	validation_1-auc:0.83318
    [46]	validation_0-auc:0.87407	validation_1-auc:0.83227
    [0]	validation_0-auc:0.79931	validation_1-auc:0.79594
    [1]	validation_0-auc:0.81987	validation_1-auc:0.81503
    [2]	validation_0-auc:0.82734	validation_1-auc:0.82126
    [3]	validation_0-auc:0.83110	validation_1-auc:0.82302
    [4]	validation_0-auc:0.83608	validation_1-auc:0.82494
    [5]	validation_0-auc:0.83914	validation_1-auc:0.83100
    [6]	validation_0-auc:0.83828	validation_1-auc:0.82999
    [7]	validation_0-auc:0.84425	validation_1-auc:0.83439
    [8]	validation_0-auc:0.84749	validation_1-auc:0.83609
    [9]	validation_0-auc:0.84727	validation_1-auc:0.83597
    [10]	validation_0-auc:0.84703	validation_1-auc:0.83250
    [11]	validation_0-auc:0.84664	validation_1-auc:0.83237
    [12]	validation_0-auc:0.85343	validation_1-auc:0.83713
    [13]	validation_0-auc:0.85671	validation_1-auc:0.83887
    [14]	validation_0-auc:0.85824	validation_1-auc:0.83919
    [15]	validation_0-auc:0.85963	validation_1-auc:0.83905
    [16]	validation_0-auc:0.86088	validation_1-auc:0.84031
    [17]	validation_0-auc:0.86214	validation_1-auc:0.84051
    [18]	validation_0-auc:0.86262	validation_1-auc:0.84051
    [19]	validation_0-auc:0.86341	validation_1-auc:0.84030
    [20]	validation_0-auc:0.86380	validation_1-auc:0.83988
    [21]	validation_0-auc:0.86411	validation_1-auc:0.84020
    [22]	validation_0-auc:0.86515	validation_1-auc:0.84033
    [23]	validation_0-auc:0.86585	validation_1-auc:0.84016
    [24]	validation_0-auc:0.86637	validation_1-auc:0.84016
    [25]	validation_0-auc:0.86687	validation_1-auc:0.83991
    [26]	validation_0-auc:0.86796	validation_1-auc:0.83979
    [27]	validation_0-auc:0.86867	validation_1-auc:0.83952
    [28]	validation_0-auc:0.86879	validation_1-auc:0.83942
    [29]	validation_0-auc:0.86907	validation_1-auc:0.83912
    [30]	validation_0-auc:0.86931	validation_1-auc:0.83907
    [31]	validation_0-auc:0.86940	validation_1-auc:0.83896
    [32]	validation_0-auc:0.87001	validation_1-auc:0.83860
    [33]	validation_0-auc:0.87017	validation_1-auc:0.83878
    [34]	validation_0-auc:0.87051	validation_1-auc:0.83830
    [35]	validation_0-auc:0.87070	validation_1-auc:0.83825
    [36]	validation_0-auc:0.87119	validation_1-auc:0.83880
    [37]	validation_0-auc:0.87126	validation_1-auc:0.83883
    [38]	validation_0-auc:0.87139	validation_1-auc:0.83882
    [39]	validation_0-auc:0.87242	validation_1-auc:0.83833
    [40]	validation_0-auc:0.87262	validation_1-auc:0.83813
    [41]	validation_0-auc:0.87276	validation_1-auc:0.83811
    [42]	validation_0-auc:0.87353	validation_1-auc:0.83806
    [43]	validation_0-auc:0.87368	validation_1-auc:0.83815
    [44]	validation_0-auc:0.87380	validation_1-auc:0.83807
    [45]	validation_0-auc:0.87392	validation_1-auc:0.83813
    [46]	validation_0-auc:0.87444	validation_1-auc:0.83757
    [0]	validation_0-auc:0.80248	validation_1-auc:0.80001
    [1]	validation_0-auc:0.82249	validation_1-auc:0.81765
    [2]	validation_0-auc:0.82833	validation_1-auc:0.82524
    [3]	validation_0-auc:0.83371	validation_1-auc:0.82814
    [4]	validation_0-auc:0.83653	validation_1-auc:0.82856
    [5]	validation_0-auc:0.83838	validation_1-auc:0.83345
    [6]	validation_0-auc:0.83823	validation_1-auc:0.83165
    [7]	validation_0-auc:0.84386	validation_1-auc:0.83505
    [8]	validation_0-auc:0.84688	validation_1-auc:0.83507
    [9]	validation_0-auc:0.84634	validation_1-auc:0.83483
    [10]	validation_0-auc:0.84564	validation_1-auc:0.83324
    [11]	validation_0-auc:0.84501	validation_1-auc:0.83283
    [12]	validation_0-auc:0.85011	validation_1-auc:0.83693
    [13]	validation_0-auc:0.85299	validation_1-auc:0.83995
    [14]	validation_0-auc:0.85523	validation_1-auc:0.84250
    [15]	validation_0-auc:0.85609	validation_1-auc:0.84183
    [16]	validation_0-auc:0.85747	validation_1-auc:0.84319
    [17]	validation_0-auc:0.85894	validation_1-auc:0.84363
    [18]	validation_0-auc:0.85942	validation_1-auc:0.84311
    [19]	validation_0-auc:0.86102	validation_1-auc:0.84368
    [20]	validation_0-auc:0.86121	validation_1-auc:0.84367
    [21]	validation_0-auc:0.86196	validation_1-auc:0.84403
    [22]	validation_0-auc:0.86290	validation_1-auc:0.84498
    [23]	validation_0-auc:0.86385	validation_1-auc:0.84460
    [24]	validation_0-auc:0.86453	validation_1-auc:0.84460
    [25]	validation_0-auc:0.86537	validation_1-auc:0.84480
    [26]	validation_0-auc:0.86586	validation_1-auc:0.84441
    [27]	validation_0-auc:0.86656	validation_1-auc:0.84401
    [28]	validation_0-auc:0.86698	validation_1-auc:0.84422
    [29]	validation_0-auc:0.86770	validation_1-auc:0.84385
    [30]	validation_0-auc:0.86778	validation_1-auc:0.84407
    [31]	validation_0-auc:0.86804	validation_1-auc:0.84395
    [32]	validation_0-auc:0.86828	validation_1-auc:0.84381
    [33]	validation_0-auc:0.86865	validation_1-auc:0.84417
    [34]	validation_0-auc:0.86902	validation_1-auc:0.84385
    [35]	validation_0-auc:0.86959	validation_1-auc:0.84369
    [36]	validation_0-auc:0.87019	validation_1-auc:0.84297
    [37]	validation_0-auc:0.87050	validation_1-auc:0.84278
    [38]	validation_0-auc:0.87180	validation_1-auc:0.84286
    [39]	validation_0-auc:0.87270	validation_1-auc:0.84224
    [40]	validation_0-auc:0.87288	validation_1-auc:0.84197
    [41]	validation_0-auc:0.87293	validation_1-auc:0.84175
    [42]	validation_0-auc:0.87421	validation_1-auc:0.84148
    [43]	validation_0-auc:0.87433	validation_1-auc:0.84121
    [44]	validation_0-auc:0.87441	validation_1-auc:0.84127
    [45]	validation_0-auc:0.87459	validation_1-auc:0.84103
    [46]	validation_0-auc:0.87477	validation_1-auc:0.84119
    [47]	validation_0-auc:0.87530	validation_1-auc:0.84128
    [48]	validation_0-auc:0.87555	validation_1-auc:0.84050
    [49]	validation_0-auc:0.87574	validation_1-auc:0.84039
    [50]	validation_0-auc:0.87578	validation_1-auc:0.84062
    [51]	validation_0-auc:0.87610	validation_1-auc:0.84105
    [0]	validation_0-auc:0.80843	validation_1-auc:0.80885
    [1]	validation_0-auc:0.82920	validation_1-auc:0.82211
    [2]	validation_0-auc:0.83320	validation_1-auc:0.82400
    [3]	validation_0-auc:0.83625	validation_1-auc:0.82577
    [4]	validation_0-auc:0.84188	validation_1-auc:0.82897
    [5]	validation_0-auc:0.84455	validation_1-auc:0.83377
    [6]	validation_0-auc:0.84503	validation_1-auc:0.82916
    [7]	validation_0-auc:0.85319	validation_1-auc:0.83364
    [8]	validation_0-auc:0.85976	validation_1-auc:0.83390
    [9]	validation_0-auc:0.85952	validation_1-auc:0.82834
    [10]	validation_0-auc:0.85919	validation_1-auc:0.82378
    [11]	validation_0-auc:0.85956	validation_1-auc:0.82400
    [12]	validation_0-auc:0.86574	validation_1-auc:0.82888
    [13]	validation_0-auc:0.87028	validation_1-auc:0.83251
    [14]	validation_0-auc:0.87240	validation_1-auc:0.83311
    [15]	validation_0-auc:0.87366	validation_1-auc:0.83080
    [16]	validation_0-auc:0.87568	validation_1-auc:0.83134
    [17]	validation_0-auc:0.87777	validation_1-auc:0.83255
    [18]	validation_0-auc:0.87906	validation_1-auc:0.83149
    [19]	validation_0-auc:0.88037	validation_1-auc:0.83083
    [20]	validation_0-auc:0.88105	validation_1-auc:0.82964
    [21]	validation_0-auc:0.88159	validation_1-auc:0.82802
    [22]	validation_0-auc:0.88227	validation_1-auc:0.82806
    [23]	validation_0-auc:0.88253	validation_1-auc:0.82806
    [24]	validation_0-auc:0.88325	validation_1-auc:0.82840
    [25]	validation_0-auc:0.88353	validation_1-auc:0.82851
    [26]	validation_0-auc:0.88385	validation_1-auc:0.82899
    [27]	validation_0-auc:0.88513	validation_1-auc:0.82988
    [28]	validation_0-auc:0.88544	validation_1-auc:0.82886
    [29]	validation_0-auc:0.88574	validation_1-auc:0.82922
    [30]	validation_0-auc:0.88591	validation_1-auc:0.82962
    [31]	validation_0-auc:0.88683	validation_1-auc:0.82951
    [32]	validation_0-auc:0.88755	validation_1-auc:0.82858
    [33]	validation_0-auc:0.88763	validation_1-auc:0.82843
    [34]	validation_0-auc:0.88790	validation_1-auc:0.82804
    [35]	validation_0-auc:0.88873	validation_1-auc:0.82692
    [36]	validation_0-auc:0.88873	validation_1-auc:0.82609
    [37]	validation_0-auc:0.88907	validation_1-auc:0.82607
    [0]	validation_0-auc:0.81304	validation_1-auc:0.81746
    [1]	validation_0-auc:0.82882	validation_1-auc:0.82026
    [2]	validation_0-auc:0.83609	validation_1-auc:0.82474
    [3]	validation_0-auc:0.84041	validation_1-auc:0.82824
    [4]	validation_0-auc:0.84760	validation_1-auc:0.83130
    [5]	validation_0-auc:0.84938	validation_1-auc:0.83590
    [6]	validation_0-auc:0.85116	validation_1-auc:0.83167
    [7]	validation_0-auc:0.85828	validation_1-auc:0.83471
    [8]	validation_0-auc:0.86371	validation_1-auc:0.83640
    [9]	validation_0-auc:0.86365	validation_1-auc:0.83549
    [10]	validation_0-auc:0.86396	validation_1-auc:0.83127
    [11]	validation_0-auc:0.86436	validation_1-auc:0.82983
    [12]	validation_0-auc:0.87068	validation_1-auc:0.83421
    [13]	validation_0-auc:0.87544	validation_1-auc:0.83773
    [14]	validation_0-auc:0.87777	validation_1-auc:0.83843
    [15]	validation_0-auc:0.87892	validation_1-auc:0.83628
    [16]	validation_0-auc:0.88034	validation_1-auc:0.83878
    [17]	validation_0-auc:0.88225	validation_1-auc:0.83749
    [18]	validation_0-auc:0.88363	validation_1-auc:0.83710
    [19]	validation_0-auc:0.88530	validation_1-auc:0.83727
    [20]	validation_0-auc:0.88610	validation_1-auc:0.83670
    [21]	validation_0-auc:0.88674	validation_1-auc:0.83629
    [22]	validation_0-auc:0.88793	validation_1-auc:0.83586
    [23]	validation_0-auc:0.88875	validation_1-auc:0.83562
    [24]	validation_0-auc:0.88914	validation_1-auc:0.83589
    [25]	validation_0-auc:0.88933	validation_1-auc:0.83575
    [26]	validation_0-auc:0.89053	validation_1-auc:0.83424
    [27]	validation_0-auc:0.89118	validation_1-auc:0.83427
    [28]	validation_0-auc:0.89173	validation_1-auc:0.83384
    [29]	validation_0-auc:0.89238	validation_1-auc:0.83318
    [30]	validation_0-auc:0.89256	validation_1-auc:0.83224
    [31]	validation_0-auc:0.89291	validation_1-auc:0.83214
    [32]	validation_0-auc:0.89357	validation_1-auc:0.83111
    [33]	validation_0-auc:0.89395	validation_1-auc:0.83114
    [34]	validation_0-auc:0.89474	validation_1-auc:0.83121
    [35]	validation_0-auc:0.89541	validation_1-auc:0.83133
    [36]	validation_0-auc:0.89590	validation_1-auc:0.83039
    [37]	validation_0-auc:0.89617	validation_1-auc:0.83024
    [38]	validation_0-auc:0.89738	validation_1-auc:0.82952
    [39]	validation_0-auc:0.89746	validation_1-auc:0.82950
    [40]	validation_0-auc:0.89751	validation_1-auc:0.82932
    [41]	validation_0-auc:0.89820	validation_1-auc:0.82838
    [42]	validation_0-auc:0.89834	validation_1-auc:0.82849
    [43]	validation_0-auc:0.89844	validation_1-auc:0.82827
    [44]	validation_0-auc:0.89908	validation_1-auc:0.82824
    [45]	validation_0-auc:0.89916	validation_1-auc:0.82788
    [0]	validation_0-auc:0.81393	validation_1-auc:0.81377
    [1]	validation_0-auc:0.82962	validation_1-auc:0.82668
    [2]	validation_0-auc:0.83724	validation_1-auc:0.83017
    [3]	validation_0-auc:0.84075	validation_1-auc:0.83079
    [4]	validation_0-auc:0.84691	validation_1-auc:0.83337
    [5]	validation_0-auc:0.84896	validation_1-auc:0.83502
    [6]	validation_0-auc:0.84980	validation_1-auc:0.82858
    [7]	validation_0-auc:0.85918	validation_1-auc:0.83358
    [8]	validation_0-auc:0.86284	validation_1-auc:0.83470
    [9]	validation_0-auc:0.86364	validation_1-auc:0.83427
    [10]	validation_0-auc:0.86242	validation_1-auc:0.83264
    [11]	validation_0-auc:0.86248	validation_1-auc:0.83255
    [12]	validation_0-auc:0.86970	validation_1-auc:0.83531
    [13]	validation_0-auc:0.87453	validation_1-auc:0.83774
    [14]	validation_0-auc:0.87632	validation_1-auc:0.83936
    [15]	validation_0-auc:0.87825	validation_1-auc:0.83676
    [16]	validation_0-auc:0.87989	validation_1-auc:0.83852
    [17]	validation_0-auc:0.88289	validation_1-auc:0.83811
    [18]	validation_0-auc:0.88337	validation_1-auc:0.83735
    [19]	validation_0-auc:0.88504	validation_1-auc:0.83720
    [20]	validation_0-auc:0.88527	validation_1-auc:0.83718
    [21]	validation_0-auc:0.88547	validation_1-auc:0.83646
    [22]	validation_0-auc:0.88633	validation_1-auc:0.83706
    [23]	validation_0-auc:0.88770	validation_1-auc:0.83714
    [24]	validation_0-auc:0.88866	validation_1-auc:0.83742
    [25]	validation_0-auc:0.88907	validation_1-auc:0.83753
    [26]	validation_0-auc:0.89067	validation_1-auc:0.83634
    [27]	validation_0-auc:0.89161	validation_1-auc:0.83565
    [28]	validation_0-auc:0.89214	validation_1-auc:0.83460
    [29]	validation_0-auc:0.89341	validation_1-auc:0.83413
    [30]	validation_0-auc:0.89378	validation_1-auc:0.83373
    [31]	validation_0-auc:0.89393	validation_1-auc:0.83396
    [32]	validation_0-auc:0.89409	validation_1-auc:0.83435
    [33]	validation_0-auc:0.89414	validation_1-auc:0.83412
    [34]	validation_0-auc:0.89433	validation_1-auc:0.83386
    [35]	validation_0-auc:0.89511	validation_1-auc:0.83338
    [36]	validation_0-auc:0.89554	validation_1-auc:0.83232
    [37]	validation_0-auc:0.89588	validation_1-auc:0.83223
    [38]	validation_0-auc:0.89608	validation_1-auc:0.83222
    [39]	validation_0-auc:0.89640	validation_1-auc:0.83187
    [40]	validation_0-auc:0.89660	validation_1-auc:0.83146
    [41]	validation_0-auc:0.89659	validation_1-auc:0.83131
    [42]	validation_0-auc:0.89788	validation_1-auc:0.83068
    [43]	validation_0-auc:0.89793	validation_1-auc:0.83069
    [0]	validation_0-auc:0.80901	validation_1-auc:0.80653
    [1]	validation_0-auc:0.82713	validation_1-auc:0.82150
    [2]	validation_0-auc:0.83227	validation_1-auc:0.82513
    [3]	validation_0-auc:0.83319	validation_1-auc:0.82525
    [4]	validation_0-auc:0.83786	validation_1-auc:0.82805
    [5]	validation_0-auc:0.84104	validation_1-auc:0.82979
    [6]	validation_0-auc:0.84432	validation_1-auc:0.82639
    [7]	validation_0-auc:0.85301	validation_1-auc:0.83411
    [8]	validation_0-auc:0.85882	validation_1-auc:0.83754
    [9]	validation_0-auc:0.85839	validation_1-auc:0.83437
    [10]	validation_0-auc:0.85606	validation_1-auc:0.83252
    [11]	validation_0-auc:0.85676	validation_1-auc:0.83031
    [12]	validation_0-auc:0.86255	validation_1-auc:0.83311
    [13]	validation_0-auc:0.86711	validation_1-auc:0.83500
    [14]	validation_0-auc:0.86926	validation_1-auc:0.83593
    [15]	validation_0-auc:0.87030	validation_1-auc:0.83404
    [16]	validation_0-auc:0.87118	validation_1-auc:0.83472
    [17]	validation_0-auc:0.87275	validation_1-auc:0.83454
    [18]	validation_0-auc:0.87366	validation_1-auc:0.83418
    [19]	validation_0-auc:0.87497	validation_1-auc:0.83324
    [20]	validation_0-auc:0.87502	validation_1-auc:0.83267
    [21]	validation_0-auc:0.87528	validation_1-auc:0.83259
    [22]	validation_0-auc:0.87571	validation_1-auc:0.83274
    [23]	validation_0-auc:0.87659	validation_1-auc:0.83362
    [24]	validation_0-auc:0.87708	validation_1-auc:0.83315
    [25]	validation_0-auc:0.87741	validation_1-auc:0.83338
    [26]	validation_0-auc:0.87761	validation_1-auc:0.83358
    [27]	validation_0-auc:0.87814	validation_1-auc:0.83337
    [28]	validation_0-auc:0.87820	validation_1-auc:0.83346
    [29]	validation_0-auc:0.87882	validation_1-auc:0.83331
    [30]	validation_0-auc:0.87900	validation_1-auc:0.83315
    [31]	validation_0-auc:0.87990	validation_1-auc:0.83277
    [32]	validation_0-auc:0.88062	validation_1-auc:0.83284
    [33]	validation_0-auc:0.88094	validation_1-auc:0.83339
    [34]	validation_0-auc:0.88210	validation_1-auc:0.83309
    [35]	validation_0-auc:0.88208	validation_1-auc:0.83317
    [36]	validation_0-auc:0.88225	validation_1-auc:0.83314
    [37]	validation_0-auc:0.88238	validation_1-auc:0.83292
    [0]	validation_0-auc:0.81176	validation_1-auc:0.80947
    [1]	validation_0-auc:0.82651	validation_1-auc:0.82286
    [2]	validation_0-auc:0.83551	validation_1-auc:0.82712
    [3]	validation_0-auc:0.83820	validation_1-auc:0.82810
    [4]	validation_0-auc:0.84733	validation_1-auc:0.82952
    [5]	validation_0-auc:0.84903	validation_1-auc:0.83409
    [6]	validation_0-auc:0.84836	validation_1-auc:0.83191
    [7]	validation_0-auc:0.85387	validation_1-auc:0.83486
    [8]	validation_0-auc:0.85876	validation_1-auc:0.83709
    [9]	validation_0-auc:0.85840	validation_1-auc:0.83730
    [10]	validation_0-auc:0.85787	validation_1-auc:0.83417
    [11]	validation_0-auc:0.85814	validation_1-auc:0.83328
    [12]	validation_0-auc:0.86432	validation_1-auc:0.83684
    [13]	validation_0-auc:0.86879	validation_1-auc:0.83901
    [14]	validation_0-auc:0.87120	validation_1-auc:0.83987
    [15]	validation_0-auc:0.87269	validation_1-auc:0.83789
    [16]	validation_0-auc:0.87454	validation_1-auc:0.83903
    [17]	validation_0-auc:0.87644	validation_1-auc:0.83873
    [18]	validation_0-auc:0.87723	validation_1-auc:0.83908
    [19]	validation_0-auc:0.87799	validation_1-auc:0.83966
    [20]	validation_0-auc:0.87881	validation_1-auc:0.83958
    [21]	validation_0-auc:0.87899	validation_1-auc:0.83960
    [22]	validation_0-auc:0.87950	validation_1-auc:0.83985
    [23]	validation_0-auc:0.88045	validation_1-auc:0.83903
    [24]	validation_0-auc:0.88121	validation_1-auc:0.83938
    [25]	validation_0-auc:0.88186	validation_1-auc:0.83941
    [26]	validation_0-auc:0.88275	validation_1-auc:0.83943
    [27]	validation_0-auc:0.88427	validation_1-auc:0.83947
    [28]	validation_0-auc:0.88443	validation_1-auc:0.83972
    [29]	validation_0-auc:0.88479	validation_1-auc:0.83903
    [30]	validation_0-auc:0.88560	validation_1-auc:0.83956
    [31]	validation_0-auc:0.88553	validation_1-auc:0.83942
    [32]	validation_0-auc:0.88569	validation_1-auc:0.83903
    [33]	validation_0-auc:0.88595	validation_1-auc:0.83902
    [34]	validation_0-auc:0.88636	validation_1-auc:0.83882
    [35]	validation_0-auc:0.88641	validation_1-auc:0.83890
    [36]	validation_0-auc:0.88709	validation_1-auc:0.83877
    [37]	validation_0-auc:0.88741	validation_1-auc:0.83862
    [38]	validation_0-auc:0.88753	validation_1-auc:0.83835
    [39]	validation_0-auc:0.88780	validation_1-auc:0.83760
    [40]	validation_0-auc:0.88771	validation_1-auc:0.83781
    [41]	validation_0-auc:0.88795	validation_1-auc:0.83789
    [42]	validation_0-auc:0.88804	validation_1-auc:0.83796
    [43]	validation_0-auc:0.88865	validation_1-auc:0.83769
    [0]	validation_0-auc:0.81519	validation_1-auc:0.81115
    [1]	validation_0-auc:0.83201	validation_1-auc:0.82366
    [2]	validation_0-auc:0.83718	validation_1-auc:0.83029
    [3]	validation_0-auc:0.84145	validation_1-auc:0.83163
    [4]	validation_0-auc:0.84628	validation_1-auc:0.83410
    [5]	validation_0-auc:0.84792	validation_1-auc:0.83694
    [6]	validation_0-auc:0.84780	validation_1-auc:0.83116
    [7]	validation_0-auc:0.85600	validation_1-auc:0.83759
    [8]	validation_0-auc:0.85905	validation_1-auc:0.83700
    [9]	validation_0-auc:0.85860	validation_1-auc:0.83638
    [10]	validation_0-auc:0.85874	validation_1-auc:0.83594
    [11]	validation_0-auc:0.85922	validation_1-auc:0.83691
    [12]	validation_0-auc:0.86559	validation_1-auc:0.84075
    [13]	validation_0-auc:0.86940	validation_1-auc:0.84350
    [14]	validation_0-auc:0.87102	validation_1-auc:0.84520
    [15]	validation_0-auc:0.87174	validation_1-auc:0.84423
    [16]	validation_0-auc:0.87351	validation_1-auc:0.84460
    [17]	validation_0-auc:0.87528	validation_1-auc:0.84395
    [18]	validation_0-auc:0.87591	validation_1-auc:0.84331
    [19]	validation_0-auc:0.87736	validation_1-auc:0.84275
    [20]	validation_0-auc:0.87771	validation_1-auc:0.84252
    [21]	validation_0-auc:0.87823	validation_1-auc:0.84160
    [22]	validation_0-auc:0.87993	validation_1-auc:0.84207
    [23]	validation_0-auc:0.88090	validation_1-auc:0.84223
    [24]	validation_0-auc:0.88140	validation_1-auc:0.84238
    [25]	validation_0-auc:0.88186	validation_1-auc:0.84258
    [26]	validation_0-auc:0.88259	validation_1-auc:0.84240
    [27]	validation_0-auc:0.88360	validation_1-auc:0.84183
    [28]	validation_0-auc:0.88403	validation_1-auc:0.84147
    [29]	validation_0-auc:0.88417	validation_1-auc:0.84140
    [30]	validation_0-auc:0.88457	validation_1-auc:0.84080
    [31]	validation_0-auc:0.88542	validation_1-auc:0.84070
    [32]	validation_0-auc:0.88561	validation_1-auc:0.84055
    [33]	validation_0-auc:0.88609	validation_1-auc:0.84024
    [34]	validation_0-auc:0.88632	validation_1-auc:0.83977
    [35]	validation_0-auc:0.88638	validation_1-auc:0.83959
    [36]	validation_0-auc:0.88644	validation_1-auc:0.83935
    [37]	validation_0-auc:0.88730	validation_1-auc:0.83898
    [38]	validation_0-auc:0.88803	validation_1-auc:0.83814
    [39]	validation_0-auc:0.88817	validation_1-auc:0.83806
    [40]	validation_0-auc:0.88815	validation_1-auc:0.83811
    [41]	validation_0-auc:0.88838	validation_1-auc:0.83807
    [42]	validation_0-auc:0.88881	validation_1-auc:0.83753
    [43]	validation_0-auc:0.88902	validation_1-auc:0.83781
    [0]	validation_0-auc:0.81007	validation_1-auc:0.80693
    [1]	validation_0-auc:0.82137	validation_1-auc:0.81877
    [2]	validation_0-auc:0.82976	validation_1-auc:0.82498
    [3]	validation_0-auc:0.83120	validation_1-auc:0.82212
    [4]	validation_0-auc:0.83382	validation_1-auc:0.82481
    [5]	validation_0-auc:0.83696	validation_1-auc:0.82672
    [6]	validation_0-auc:0.83976	validation_1-auc:0.83016
    [7]	validation_0-auc:0.84177	validation_1-auc:0.83330
    [8]	validation_0-auc:0.84585	validation_1-auc:0.83282
    [9]	validation_0-auc:0.84984	validation_1-auc:0.83519
    [10]	validation_0-auc:0.85146	validation_1-auc:0.83530
    [11]	validation_0-auc:0.85113	validation_1-auc:0.83380
    [12]	validation_0-auc:0.85502	validation_1-auc:0.83622
    [13]	validation_0-auc:0.85797	validation_1-auc:0.83644
    [14]	validation_0-auc:0.85990	validation_1-auc:0.83686
    [15]	validation_0-auc:0.86114	validation_1-auc:0.83639
    [16]	validation_0-auc:0.86159	validation_1-auc:0.83602
    [17]	validation_0-auc:0.86284	validation_1-auc:0.83501
    [18]	validation_0-auc:0.86405	validation_1-auc:0.83454
    [19]	validation_0-auc:0.86498	validation_1-auc:0.83497
    [20]	validation_0-auc:0.86595	validation_1-auc:0.83417
    [21]	validation_0-auc:0.86759	validation_1-auc:0.83454
    [22]	validation_0-auc:0.86808	validation_1-auc:0.83466
    [23]	validation_0-auc:0.86829	validation_1-auc:0.83461
    [24]	validation_0-auc:0.86858	validation_1-auc:0.83422
    [25]	validation_0-auc:0.86943	validation_1-auc:0.83371
    [26]	validation_0-auc:0.86987	validation_1-auc:0.83392
    [27]	validation_0-auc:0.87053	validation_1-auc:0.83330
    [28]	validation_0-auc:0.87106	validation_1-auc:0.83367
    [29]	validation_0-auc:0.87112	validation_1-auc:0.83371
    [30]	validation_0-auc:0.87153	validation_1-auc:0.83435
    [31]	validation_0-auc:0.87180	validation_1-auc:0.83437
    [32]	validation_0-auc:0.87290	validation_1-auc:0.83459
    [33]	validation_0-auc:0.87307	validation_1-auc:0.83470
    [34]	validation_0-auc:0.87350	validation_1-auc:0.83407
    [35]	validation_0-auc:0.87396	validation_1-auc:0.83319
    [36]	validation_0-auc:0.87465	validation_1-auc:0.83300
    [37]	validation_0-auc:0.87470	validation_1-auc:0.83311
    [38]	validation_0-auc:0.87500	validation_1-auc:0.83281
    [39]	validation_0-auc:0.87592	validation_1-auc:0.83273
    [40]	validation_0-auc:0.87624	validation_1-auc:0.83299
    [41]	validation_0-auc:0.87747	validation_1-auc:0.83274
    [42]	validation_0-auc:0.87753	validation_1-auc:0.83254
    [43]	validation_0-auc:0.87845	validation_1-auc:0.83286
    [0]	validation_0-auc:0.80863	validation_1-auc:0.80010
    [1]	validation_0-auc:0.82349	validation_1-auc:0.81717
    [2]	validation_0-auc:0.82654	validation_1-auc:0.81737
    [3]	validation_0-auc:0.82988	validation_1-auc:0.82281
    [4]	validation_0-auc:0.83570	validation_1-auc:0.82554
    [5]	validation_0-auc:0.83917	validation_1-auc:0.82930
    [6]	validation_0-auc:0.84492	validation_1-auc:0.83396
    [7]	validation_0-auc:0.84657	validation_1-auc:0.83569
    [8]	validation_0-auc:0.84837	validation_1-auc:0.83476
    [9]	validation_0-auc:0.85009	validation_1-auc:0.83841
    [10]	validation_0-auc:0.85017	validation_1-auc:0.83887
    [11]	validation_0-auc:0.85091	validation_1-auc:0.83723
    [12]	validation_0-auc:0.85584	validation_1-auc:0.83976
    [13]	validation_0-auc:0.85900	validation_1-auc:0.84063
    [14]	validation_0-auc:0.86060	validation_1-auc:0.84054
    [15]	validation_0-auc:0.86167	validation_1-auc:0.84086
    [16]	validation_0-auc:0.86304	validation_1-auc:0.84085
    [17]	validation_0-auc:0.86382	validation_1-auc:0.83947
    [18]	validation_0-auc:0.86463	validation_1-auc:0.83971
    [19]	validation_0-auc:0.86560	validation_1-auc:0.84059
    [20]	validation_0-auc:0.86649	validation_1-auc:0.83981
    [21]	validation_0-auc:0.86761	validation_1-auc:0.84030
    [22]	validation_0-auc:0.86864	validation_1-auc:0.84050
    [23]	validation_0-auc:0.86914	validation_1-auc:0.83978
    [24]	validation_0-auc:0.86953	validation_1-auc:0.84033
    [25]	validation_0-auc:0.86991	validation_1-auc:0.84000
    [26]	validation_0-auc:0.87003	validation_1-auc:0.83998
    [27]	validation_0-auc:0.87113	validation_1-auc:0.83964
    [28]	validation_0-auc:0.87205	validation_1-auc:0.83972
    [29]	validation_0-auc:0.87326	validation_1-auc:0.83984
    [30]	validation_0-auc:0.87359	validation_1-auc:0.83929
    [31]	validation_0-auc:0.87367	validation_1-auc:0.83938
    [32]	validation_0-auc:0.87436	validation_1-auc:0.83918
    [33]	validation_0-auc:0.87490	validation_1-auc:0.83990
    [34]	validation_0-auc:0.87594	validation_1-auc:0.84011
    [35]	validation_0-auc:0.87619	validation_1-auc:0.83988
    [36]	validation_0-auc:0.87647	validation_1-auc:0.83991
    [37]	validation_0-auc:0.87657	validation_1-auc:0.83991
    [38]	validation_0-auc:0.87676	validation_1-auc:0.83987
    [39]	validation_0-auc:0.87702	validation_1-auc:0.83973
    [40]	validation_0-auc:0.87711	validation_1-auc:0.83990
    [41]	validation_0-auc:0.87731	validation_1-auc:0.83941
    [42]	validation_0-auc:0.87786	validation_1-auc:0.83934
    [43]	validation_0-auc:0.87815	validation_1-auc:0.83924
    [44]	validation_0-auc:0.87850	validation_1-auc:0.83882
    [0]	validation_0-auc:0.82005	validation_1-auc:0.81815
    [1]	validation_0-auc:0.82547	validation_1-auc:0.82159
    [2]	validation_0-auc:0.83019	validation_1-auc:0.82631
    [3]	validation_0-auc:0.83230	validation_1-auc:0.82660
    [4]	validation_0-auc:0.83488	validation_1-auc:0.82988
    [5]	validation_0-auc:0.83888	validation_1-auc:0.83262
    [6]	validation_0-auc:0.84242	validation_1-auc:0.83408
    [7]	validation_0-auc:0.84581	validation_1-auc:0.83560
    [8]	validation_0-auc:0.84775	validation_1-auc:0.83617
    [9]	validation_0-auc:0.84989	validation_1-auc:0.83746
    [10]	validation_0-auc:0.85052	validation_1-auc:0.83816
    [11]	validation_0-auc:0.84982	validation_1-auc:0.83603
    [12]	validation_0-auc:0.85408	validation_1-auc:0.83825
    [13]	validation_0-auc:0.85547	validation_1-auc:0.83955
    [14]	validation_0-auc:0.85818	validation_1-auc:0.84292
    [15]	validation_0-auc:0.85990	validation_1-auc:0.84361
    [16]	validation_0-auc:0.86142	validation_1-auc:0.84287
    [17]	validation_0-auc:0.86246	validation_1-auc:0.84280
    [18]	validation_0-auc:0.86276	validation_1-auc:0.84297
    [19]	validation_0-auc:0.86367	validation_1-auc:0.84290
    [20]	validation_0-auc:0.86489	validation_1-auc:0.84279
    [21]	validation_0-auc:0.86540	validation_1-auc:0.84307
    [22]	validation_0-auc:0.86631	validation_1-auc:0.84285
    [23]	validation_0-auc:0.86687	validation_1-auc:0.84289
    [24]	validation_0-auc:0.86776	validation_1-auc:0.84289
    [25]	validation_0-auc:0.86829	validation_1-auc:0.84279
    [26]	validation_0-auc:0.86862	validation_1-auc:0.84237
    [27]	validation_0-auc:0.87011	validation_1-auc:0.84232
    [28]	validation_0-auc:0.87063	validation_1-auc:0.84224
    [29]	validation_0-auc:0.87064	validation_1-auc:0.84199
    [30]	validation_0-auc:0.87109	validation_1-auc:0.84246
    [31]	validation_0-auc:0.87189	validation_1-auc:0.84252
    [32]	validation_0-auc:0.87276	validation_1-auc:0.84147
    [33]	validation_0-auc:0.87303	validation_1-auc:0.84149
    [34]	validation_0-auc:0.87350	validation_1-auc:0.84118
    [35]	validation_0-auc:0.87370	validation_1-auc:0.84115
    [36]	validation_0-auc:0.87408	validation_1-auc:0.84113
    [37]	validation_0-auc:0.87477	validation_1-auc:0.84038
    [38]	validation_0-auc:0.87530	validation_1-auc:0.84009
    [39]	validation_0-auc:0.87541	validation_1-auc:0.83988
    [40]	validation_0-auc:0.87556	validation_1-auc:0.83984
    [41]	validation_0-auc:0.87580	validation_1-auc:0.83991
    [42]	validation_0-auc:0.87631	validation_1-auc:0.83942
    [43]	validation_0-auc:0.87667	validation_1-auc:0.83926
    [44]	validation_0-auc:0.87715	validation_1-auc:0.83916
    [0]	validation_0-auc:0.81105	validation_1-auc:0.80637
    [1]	validation_0-auc:0.82008	validation_1-auc:0.81881
    [2]	validation_0-auc:0.82922	validation_1-auc:0.82532
    [3]	validation_0-auc:0.83159	validation_1-auc:0.82594
    [4]	validation_0-auc:0.83378	validation_1-auc:0.82618
    [5]	validation_0-auc:0.83671	validation_1-auc:0.82887
    [6]	validation_0-auc:0.84111	validation_1-auc:0.83302
    [7]	validation_0-auc:0.84227	validation_1-auc:0.83380
    [8]	validation_0-auc:0.84423	validation_1-auc:0.83346
    [9]	validation_0-auc:0.84742	validation_1-auc:0.83581
    [10]	validation_0-auc:0.84984	validation_1-auc:0.83563
    [11]	validation_0-auc:0.84933	validation_1-auc:0.83344
    [12]	validation_0-auc:0.85285	validation_1-auc:0.83653
    [13]	validation_0-auc:0.85493	validation_1-auc:0.83796
    [14]	validation_0-auc:0.85653	validation_1-auc:0.83880
    [15]	validation_0-auc:0.85803	validation_1-auc:0.83841
    [16]	validation_0-auc:0.85923	validation_1-auc:0.83773
    [17]	validation_0-auc:0.85983	validation_1-auc:0.83709
    [18]	validation_0-auc:0.86163	validation_1-auc:0.83622
    [19]	validation_0-auc:0.86232	validation_1-auc:0.83513
    [20]	validation_0-auc:0.86288	validation_1-auc:0.83518
    [21]	validation_0-auc:0.86375	validation_1-auc:0.83543
    [22]	validation_0-auc:0.86417	validation_1-auc:0.83540
    [23]	validation_0-auc:0.86462	validation_1-auc:0.83510
    [24]	validation_0-auc:0.86485	validation_1-auc:0.83477
    [25]	validation_0-auc:0.86529	validation_1-auc:0.83484
    [26]	validation_0-auc:0.86549	validation_1-auc:0.83473
    [27]	validation_0-auc:0.86570	validation_1-auc:0.83481
    [28]	validation_0-auc:0.86580	validation_1-auc:0.83485
    [29]	validation_0-auc:0.86653	validation_1-auc:0.83501
    [30]	validation_0-auc:0.86667	validation_1-auc:0.83465
    [31]	validation_0-auc:0.86791	validation_1-auc:0.83486
    [32]	validation_0-auc:0.86803	validation_1-auc:0.83488
    [33]	validation_0-auc:0.86812	validation_1-auc:0.83473
    [34]	validation_0-auc:0.86820	validation_1-auc:0.83483
    [35]	validation_0-auc:0.86827	validation_1-auc:0.83508
    [36]	validation_0-auc:0.86861	validation_1-auc:0.83435
    [37]	validation_0-auc:0.86866	validation_1-auc:0.83425
    [38]	validation_0-auc:0.86891	validation_1-auc:0.83451
    [39]	validation_0-auc:0.86913	validation_1-auc:0.83425
    [40]	validation_0-auc:0.86939	validation_1-auc:0.83430
    [41]	validation_0-auc:0.86942	validation_1-auc:0.83443
    [42]	validation_0-auc:0.86954	validation_1-auc:0.83436
    [43]	validation_0-auc:0.87019	validation_1-auc:0.83441
    [0]	validation_0-auc:0.81067	validation_1-auc:0.81109
    [1]	validation_0-auc:0.82045	validation_1-auc:0.81627
    [2]	validation_0-auc:0.82760	validation_1-auc:0.82116
    [3]	validation_0-auc:0.82925	validation_1-auc:0.81730
    [4]	validation_0-auc:0.83628	validation_1-auc:0.82554
    [5]	validation_0-auc:0.83889	validation_1-auc:0.82992
    [6]	validation_0-auc:0.84258	validation_1-auc:0.83304
    [7]	validation_0-auc:0.84515	validation_1-auc:0.83327
    [8]	validation_0-auc:0.84797	validation_1-auc:0.83479
    [9]	validation_0-auc:0.84982	validation_1-auc:0.83737
    [10]	validation_0-auc:0.84996	validation_1-auc:0.83746
    [11]	validation_0-auc:0.84929	validation_1-auc:0.83715
    [12]	validation_0-auc:0.85506	validation_1-auc:0.83957
    [13]	validation_0-auc:0.85817	validation_1-auc:0.84131
    [14]	validation_0-auc:0.85946	validation_1-auc:0.84041
    [15]	validation_0-auc:0.86040	validation_1-auc:0.83984
    [16]	validation_0-auc:0.86126	validation_1-auc:0.83954
    [17]	validation_0-auc:0.86170	validation_1-auc:0.83947
    [18]	validation_0-auc:0.86276	validation_1-auc:0.83945
    [19]	validation_0-auc:0.86327	validation_1-auc:0.84019
    [20]	validation_0-auc:0.86381	validation_1-auc:0.84075
    [21]	validation_0-auc:0.86454	validation_1-auc:0.84078
    [22]	validation_0-auc:0.86531	validation_1-auc:0.84164
    [23]	validation_0-auc:0.86598	validation_1-auc:0.84128
    [24]	validation_0-auc:0.86650	validation_1-auc:0.84078
    [25]	validation_0-auc:0.86717	validation_1-auc:0.84069
    [26]	validation_0-auc:0.86742	validation_1-auc:0.84066
    [27]	validation_0-auc:0.86807	validation_1-auc:0.84017
    [28]	validation_0-auc:0.86913	validation_1-auc:0.84027
    [29]	validation_0-auc:0.86952	validation_1-auc:0.84014
    [30]	validation_0-auc:0.86972	validation_1-auc:0.84016
    [31]	validation_0-auc:0.86996	validation_1-auc:0.83992
    [32]	validation_0-auc:0.87071	validation_1-auc:0.84001
    [33]	validation_0-auc:0.87090	validation_1-auc:0.83997
    [34]	validation_0-auc:0.87110	validation_1-auc:0.83969
    [35]	validation_0-auc:0.87146	validation_1-auc:0.83964
    [36]	validation_0-auc:0.87214	validation_1-auc:0.84006
    [37]	validation_0-auc:0.87240	validation_1-auc:0.83987
    [38]	validation_0-auc:0.87260	validation_1-auc:0.83995
    [39]	validation_0-auc:0.87270	validation_1-auc:0.84021
    [40]	validation_0-auc:0.87270	validation_1-auc:0.84066
    [41]	validation_0-auc:0.87318	validation_1-auc:0.84095
    [42]	validation_0-auc:0.87367	validation_1-auc:0.84074
    [43]	validation_0-auc:0.87429	validation_1-auc:0.84057
    [44]	validation_0-auc:0.87440	validation_1-auc:0.84028
    [45]	validation_0-auc:0.87508	validation_1-auc:0.84011
    [46]	validation_0-auc:0.87550	validation_1-auc:0.83972
    [47]	validation_0-auc:0.87604	validation_1-auc:0.83880
    [48]	validation_0-auc:0.87631	validation_1-auc:0.83876
    [49]	validation_0-auc:0.87628	validation_1-auc:0.83900
    [50]	validation_0-auc:0.87638	validation_1-auc:0.83902
    [51]	validation_0-auc:0.87647	validation_1-auc:0.83930
    [52]	validation_0-auc:0.87648	validation_1-auc:0.83930
    [0]	validation_0-auc:0.81835	validation_1-auc:0.81691
    [1]	validation_0-auc:0.82862	validation_1-auc:0.82346
    [2]	validation_0-auc:0.83280	validation_1-auc:0.82893
    [3]	validation_0-auc:0.83563	validation_1-auc:0.82931
    [4]	validation_0-auc:0.83780	validation_1-auc:0.83200
    [5]	validation_0-auc:0.83975	validation_1-auc:0.83280
    [6]	validation_0-auc:0.84205	validation_1-auc:0.83374
    [7]	validation_0-auc:0.84453	validation_1-auc:0.83256
    [8]	validation_0-auc:0.84638	validation_1-auc:0.83384
    [9]	validation_0-auc:0.84986	validation_1-auc:0.83670
    [10]	validation_0-auc:0.85058	validation_1-auc:0.83825
    [11]	validation_0-auc:0.84986	validation_1-auc:0.83646
    [12]	validation_0-auc:0.85321	validation_1-auc:0.83744
    [13]	validation_0-auc:0.85479	validation_1-auc:0.83942
    [14]	validation_0-auc:0.85614	validation_1-auc:0.84091
    [15]	validation_0-auc:0.85710	validation_1-auc:0.84170
    [16]	validation_0-auc:0.85892	validation_1-auc:0.84239
    [17]	validation_0-auc:0.86024	validation_1-auc:0.84215
    [18]	validation_0-auc:0.86146	validation_1-auc:0.84247
    [19]	validation_0-auc:0.86202	validation_1-auc:0.84237
    [20]	validation_0-auc:0.86268	validation_1-auc:0.84152
    [21]	validation_0-auc:0.86343	validation_1-auc:0.84132
    [22]	validation_0-auc:0.86490	validation_1-auc:0.84044
    [23]	validation_0-auc:0.86602	validation_1-auc:0.84073
    [24]	validation_0-auc:0.86688	validation_1-auc:0.84082
    [25]	validation_0-auc:0.86778	validation_1-auc:0.84074
    [26]	validation_0-auc:0.86849	validation_1-auc:0.84076
    [27]	validation_0-auc:0.86909	validation_1-auc:0.84096
    [28]	validation_0-auc:0.86930	validation_1-auc:0.84113
    [29]	validation_0-auc:0.86973	validation_1-auc:0.84187
    [30]	validation_0-auc:0.87070	validation_1-auc:0.84167
    [31]	validation_0-auc:0.87108	validation_1-auc:0.84174
    [32]	validation_0-auc:0.87124	validation_1-auc:0.84166
    [33]	validation_0-auc:0.87154	validation_1-auc:0.84142
    [34]	validation_0-auc:0.87216	validation_1-auc:0.84153
    [35]	validation_0-auc:0.87288	validation_1-auc:0.84147
    [36]	validation_0-auc:0.87324	validation_1-auc:0.84136
    [37]	validation_0-auc:0.87343	validation_1-auc:0.84116
    [38]	validation_0-auc:0.87351	validation_1-auc:0.84114
    [39]	validation_0-auc:0.87406	validation_1-auc:0.84087
    [40]	validation_0-auc:0.87415	validation_1-auc:0.84088
    [41]	validation_0-auc:0.87540	validation_1-auc:0.84065
    [42]	validation_0-auc:0.87577	validation_1-auc:0.84078
    [43]	validation_0-auc:0.87597	validation_1-auc:0.84097
    [44]	validation_0-auc:0.87647	validation_1-auc:0.84047
    [45]	validation_0-auc:0.87667	validation_1-auc:0.84048
    [46]	validation_0-auc:0.87670	validation_1-auc:0.84016
    [47]	validation_0-auc:0.87718	validation_1-auc:0.84000
    [0]	validation_0-auc:0.81685	validation_1-auc:0.81075
    [1]	validation_0-auc:0.82791	validation_1-auc:0.82283
    [2]	validation_0-auc:0.83537	validation_1-auc:0.82615
    [3]	validation_0-auc:0.83996	validation_1-auc:0.82712
    [4]	validation_0-auc:0.84558	validation_1-auc:0.82791
    [5]	validation_0-auc:0.84781	validation_1-auc:0.82977
    [6]	validation_0-auc:0.85151	validation_1-auc:0.83373
    [7]	validation_0-auc:0.85510	validation_1-auc:0.83444
    [8]	validation_0-auc:0.85998	validation_1-auc:0.83601
    [9]	validation_0-auc:0.86238	validation_1-auc:0.83804
    [10]	validation_0-auc:0.86434	validation_1-auc:0.83584
    [11]	validation_0-auc:0.86583	validation_1-auc:0.83093
    [12]	validation_0-auc:0.87078	validation_1-auc:0.83235
    [13]	validation_0-auc:0.87454	validation_1-auc:0.83253
    [14]	validation_0-auc:0.87641	validation_1-auc:0.83254
    [15]	validation_0-auc:0.87857	validation_1-auc:0.83218
    [16]	validation_0-auc:0.87974	validation_1-auc:0.83171
    [17]	validation_0-auc:0.88123	validation_1-auc:0.83115
    [18]	validation_0-auc:0.88255	validation_1-auc:0.83119
    [19]	validation_0-auc:0.88333	validation_1-auc:0.83139
    [20]	validation_0-auc:0.88409	validation_1-auc:0.83082
    [21]	validation_0-auc:0.88508	validation_1-auc:0.83044
    [22]	validation_0-auc:0.88632	validation_1-auc:0.83025
    [23]	validation_0-auc:0.88673	validation_1-auc:0.83047
    [24]	validation_0-auc:0.88740	validation_1-auc:0.82903
    [25]	validation_0-auc:0.88770	validation_1-auc:0.82895
    [26]	validation_0-auc:0.88791	validation_1-auc:0.82913
    [27]	validation_0-auc:0.88810	validation_1-auc:0.82881
    [28]	validation_0-auc:0.88830	validation_1-auc:0.82901
    [29]	validation_0-auc:0.88830	validation_1-auc:0.82910
    [30]	validation_0-auc:0.88892	validation_1-auc:0.82854
    [31]	validation_0-auc:0.88897	validation_1-auc:0.82859
    [32]	validation_0-auc:0.88913	validation_1-auc:0.82837
    [33]	validation_0-auc:0.88934	validation_1-auc:0.82847
    [34]	validation_0-auc:0.89033	validation_1-auc:0.82891
    [35]	validation_0-auc:0.89098	validation_1-auc:0.82869
    [36]	validation_0-auc:0.89159	validation_1-auc:0.82814
    [37]	validation_0-auc:0.89167	validation_1-auc:0.82822
    [38]	validation_0-auc:0.89183	validation_1-auc:0.82764
    [0]	validation_0-auc:0.81432	validation_1-auc:0.80561
    [1]	validation_0-auc:0.82773	validation_1-auc:0.81858
    [2]	validation_0-auc:0.83521	validation_1-auc:0.82197
    [3]	validation_0-auc:0.84270	validation_1-auc:0.82752
    [4]	validation_0-auc:0.84818	validation_1-auc:0.83067
    [5]	validation_0-auc:0.85320	validation_1-auc:0.83293
    [6]	validation_0-auc:0.85966	validation_1-auc:0.83461
    [7]	validation_0-auc:0.86215	validation_1-auc:0.83669
    [8]	validation_0-auc:0.86492	validation_1-auc:0.83749
    [9]	validation_0-auc:0.86708	validation_1-auc:0.83803
    [10]	validation_0-auc:0.86784	validation_1-auc:0.83916
    [11]	validation_0-auc:0.86926	validation_1-auc:0.83507
    [12]	validation_0-auc:0.87504	validation_1-auc:0.83780
    [13]	validation_0-auc:0.87858	validation_1-auc:0.83724
    [14]	validation_0-auc:0.88060	validation_1-auc:0.83823
    [15]	validation_0-auc:0.88239	validation_1-auc:0.83917
    [16]	validation_0-auc:0.88360	validation_1-auc:0.83884
    [17]	validation_0-auc:0.88481	validation_1-auc:0.83815
    [18]	validation_0-auc:0.88594	validation_1-auc:0.83772
    [19]	validation_0-auc:0.88671	validation_1-auc:0.83853
    [20]	validation_0-auc:0.88717	validation_1-auc:0.83800
    [21]	validation_0-auc:0.88814	validation_1-auc:0.83800
    [22]	validation_0-auc:0.88867	validation_1-auc:0.83811
    [23]	validation_0-auc:0.88984	validation_1-auc:0.83780
    [24]	validation_0-auc:0.89035	validation_1-auc:0.83731
    [25]	validation_0-auc:0.89094	validation_1-auc:0.83717
    [26]	validation_0-auc:0.89143	validation_1-auc:0.83694
    [27]	validation_0-auc:0.89200	validation_1-auc:0.83660
    [28]	validation_0-auc:0.89281	validation_1-auc:0.83711
    [29]	validation_0-auc:0.89290	validation_1-auc:0.83771
    [30]	validation_0-auc:0.89310	validation_1-auc:0.83717
    [31]	validation_0-auc:0.89338	validation_1-auc:0.83704
    [32]	validation_0-auc:0.89415	validation_1-auc:0.83731
    [33]	validation_0-auc:0.89484	validation_1-auc:0.83674
    [34]	validation_0-auc:0.89584	validation_1-auc:0.83723
    [35]	validation_0-auc:0.89607	validation_1-auc:0.83724
    [36]	validation_0-auc:0.89655	validation_1-auc:0.83646
    [37]	validation_0-auc:0.89681	validation_1-auc:0.83664
    [38]	validation_0-auc:0.89694	validation_1-auc:0.83654
    [39]	validation_0-auc:0.89735	validation_1-auc:0.83618
    [40]	validation_0-auc:0.89847	validation_1-auc:0.83615
    [41]	validation_0-auc:0.89846	validation_1-auc:0.83613
    [42]	validation_0-auc:0.89855	validation_1-auc:0.83601
    [43]	validation_0-auc:0.89853	validation_1-auc:0.83532
    [44]	validation_0-auc:0.89863	validation_1-auc:0.83517
    [0]	validation_0-auc:0.82507	validation_1-auc:0.81932
    [1]	validation_0-auc:0.83077	validation_1-auc:0.82081
    [2]	validation_0-auc:0.83704	validation_1-auc:0.82829
    [3]	validation_0-auc:0.84184	validation_1-auc:0.83011
    [4]	validation_0-auc:0.84747	validation_1-auc:0.83438
    [5]	validation_0-auc:0.85320	validation_1-auc:0.83489
    [6]	validation_0-auc:0.85706	validation_1-auc:0.83487
    [7]	validation_0-auc:0.86253	validation_1-auc:0.83489
    [8]	validation_0-auc:0.86436	validation_1-auc:0.83406
    [9]	validation_0-auc:0.86657	validation_1-auc:0.83491
    [10]	validation_0-auc:0.86829	validation_1-auc:0.83443
    [11]	validation_0-auc:0.86897	validation_1-auc:0.83444
    [12]	validation_0-auc:0.87502	validation_1-auc:0.83503
    [13]	validation_0-auc:0.87822	validation_1-auc:0.83527
    [14]	validation_0-auc:0.88101	validation_1-auc:0.83735
    [15]	validation_0-auc:0.88255	validation_1-auc:0.83832
    [16]	validation_0-auc:0.88487	validation_1-auc:0.83666
    [17]	validation_0-auc:0.88665	validation_1-auc:0.83625
    [18]	validation_0-auc:0.88699	validation_1-auc:0.83559
    [19]	validation_0-auc:0.88791	validation_1-auc:0.83546
    [20]	validation_0-auc:0.88884	validation_1-auc:0.83529
    [21]	validation_0-auc:0.89002	validation_1-auc:0.83494
    [22]	validation_0-auc:0.89170	validation_1-auc:0.83342
    [23]	validation_0-auc:0.89226	validation_1-auc:0.83397
    [24]	validation_0-auc:0.89264	validation_1-auc:0.83420
    [25]	validation_0-auc:0.89307	validation_1-auc:0.83461
    [26]	validation_0-auc:0.89397	validation_1-auc:0.83459
    [27]	validation_0-auc:0.89434	validation_1-auc:0.83497
    [28]	validation_0-auc:0.89490	validation_1-auc:0.83437
    [29]	validation_0-auc:0.89493	validation_1-auc:0.83446
    [30]	validation_0-auc:0.89559	validation_1-auc:0.83426
    [31]	validation_0-auc:0.89565	validation_1-auc:0.83405
    [32]	validation_0-auc:0.89653	validation_1-auc:0.83320
    [33]	validation_0-auc:0.89680	validation_1-auc:0.83358
    [34]	validation_0-auc:0.89762	validation_1-auc:0.83320
    [35]	validation_0-auc:0.89782	validation_1-auc:0.83337
    [36]	validation_0-auc:0.89811	validation_1-auc:0.83330
    [37]	validation_0-auc:0.89813	validation_1-auc:0.83297
    [38]	validation_0-auc:0.89838	validation_1-auc:0.83265
    [39]	validation_0-auc:0.89870	validation_1-auc:0.83239
    [40]	validation_0-auc:0.89877	validation_1-auc:0.83215
    [41]	validation_0-auc:0.89897	validation_1-auc:0.83207
    [42]	validation_0-auc:0.89904	validation_1-auc:0.83192
    [43]	validation_0-auc:0.89918	validation_1-auc:0.83183
    [44]	validation_0-auc:0.89926	validation_1-auc:0.83188
    [0]	validation_0-auc:0.81664	validation_1-auc:0.81074
    [1]	validation_0-auc:0.83122	validation_1-auc:0.82620
    [2]	validation_0-auc:0.83368	validation_1-auc:0.82978
    [3]	validation_0-auc:0.83631	validation_1-auc:0.82844
    [4]	validation_0-auc:0.83978	validation_1-auc:0.82902
    [5]	validation_0-auc:0.84479	validation_1-auc:0.83150
    [6]	validation_0-auc:0.85038	validation_1-auc:0.83301
    [7]	validation_0-auc:0.85463	validation_1-auc:0.83347
    [8]	validation_0-auc:0.85806	validation_1-auc:0.83504
    [9]	validation_0-auc:0.86217	validation_1-auc:0.83665
    [10]	validation_0-auc:0.86465	validation_1-auc:0.83511
    [11]	validation_0-auc:0.86428	validation_1-auc:0.83467
    [12]	validation_0-auc:0.86831	validation_1-auc:0.83612
    [13]	validation_0-auc:0.87023	validation_1-auc:0.83695
    [14]	validation_0-auc:0.87207	validation_1-auc:0.83698
    [15]	validation_0-auc:0.87350	validation_1-auc:0.83506
    [16]	validation_0-auc:0.87495	validation_1-auc:0.83587
    [17]	validation_0-auc:0.87589	validation_1-auc:0.83520
    [18]	validation_0-auc:0.87639	validation_1-auc:0.83404
    [19]	validation_0-auc:0.87737	validation_1-auc:0.83451
    [20]	validation_0-auc:0.87782	validation_1-auc:0.83428
    [21]	validation_0-auc:0.87909	validation_1-auc:0.83329
    [22]	validation_0-auc:0.87952	validation_1-auc:0.83338
    [23]	validation_0-auc:0.88028	validation_1-auc:0.83328
    [24]	validation_0-auc:0.88042	validation_1-auc:0.83382
    [25]	validation_0-auc:0.88052	validation_1-auc:0.83414
    [26]	validation_0-auc:0.88134	validation_1-auc:0.83412
    [27]	validation_0-auc:0.88136	validation_1-auc:0.83369
    [28]	validation_0-auc:0.88150	validation_1-auc:0.83337
    [29]	validation_0-auc:0.88270	validation_1-auc:0.83232
    [30]	validation_0-auc:0.88363	validation_1-auc:0.83105
    [31]	validation_0-auc:0.88371	validation_1-auc:0.83010
    [32]	validation_0-auc:0.88401	validation_1-auc:0.82954
    [33]	validation_0-auc:0.88406	validation_1-auc:0.82942
    [34]	validation_0-auc:0.88501	validation_1-auc:0.82801
    [35]	validation_0-auc:0.88578	validation_1-auc:0.82829
    [36]	validation_0-auc:0.88589	validation_1-auc:0.82844
    [37]	validation_0-auc:0.88627	validation_1-auc:0.82817
    [38]	validation_0-auc:0.88647	validation_1-auc:0.82852
    [39]	validation_0-auc:0.88652	validation_1-auc:0.82824
    [40]	validation_0-auc:0.88661	validation_1-auc:0.82841
    [41]	validation_0-auc:0.88653	validation_1-auc:0.82791
    [42]	validation_0-auc:0.88775	validation_1-auc:0.82698
    [43]	validation_0-auc:0.88792	validation_1-auc:0.82706
    [0]	validation_0-auc:0.81435	validation_1-auc:0.81417
    [1]	validation_0-auc:0.82683	validation_1-auc:0.81889
    [2]	validation_0-auc:0.83601	validation_1-auc:0.82604
    [3]	validation_0-auc:0.83952	validation_1-auc:0.82594
    [4]	validation_0-auc:0.84601	validation_1-auc:0.83185
    [5]	validation_0-auc:0.84953	validation_1-auc:0.83241
    [6]	validation_0-auc:0.85560	validation_1-auc:0.83679
    [7]	validation_0-auc:0.85940	validation_1-auc:0.83663
    [8]	validation_0-auc:0.86252	validation_1-auc:0.83756
    [9]	validation_0-auc:0.86461	validation_1-auc:0.84151
    [10]	validation_0-auc:0.86372	validation_1-auc:0.84082
    [11]	validation_0-auc:0.86522	validation_1-auc:0.83606
    [12]	validation_0-auc:0.86931	validation_1-auc:0.83940
    [13]	validation_0-auc:0.87362	validation_1-auc:0.84092
    [14]	validation_0-auc:0.87499	validation_1-auc:0.84059
    [15]	validation_0-auc:0.87676	validation_1-auc:0.83958
    [16]	validation_0-auc:0.87752	validation_1-auc:0.84089
    [17]	validation_0-auc:0.87818	validation_1-auc:0.84054
    [18]	validation_0-auc:0.87849	validation_1-auc:0.84054
    [19]	validation_0-auc:0.87920	validation_1-auc:0.84046
    [20]	validation_0-auc:0.87945	validation_1-auc:0.83994
    [21]	validation_0-auc:0.87993	validation_1-auc:0.84012
    [22]	validation_0-auc:0.88087	validation_1-auc:0.83975
    [23]	validation_0-auc:0.88169	validation_1-auc:0.83978
    [24]	validation_0-auc:0.88276	validation_1-auc:0.84052
    [25]	validation_0-auc:0.88324	validation_1-auc:0.84057
    [26]	validation_0-auc:0.88340	validation_1-auc:0.84027
    [27]	validation_0-auc:0.88371	validation_1-auc:0.83987
    [28]	validation_0-auc:0.88387	validation_1-auc:0.83993
    [29]	validation_0-auc:0.88442	validation_1-auc:0.84004
    [30]	validation_0-auc:0.88490	validation_1-auc:0.83902
    [31]	validation_0-auc:0.88509	validation_1-auc:0.83907
    [32]	validation_0-auc:0.88554	validation_1-auc:0.83876
    [33]	validation_0-auc:0.88624	validation_1-auc:0.83852
    [34]	validation_0-auc:0.88711	validation_1-auc:0.83824
    [35]	validation_0-auc:0.88740	validation_1-auc:0.83792
    [36]	validation_0-auc:0.88756	validation_1-auc:0.83799
    [37]	validation_0-auc:0.88854	validation_1-auc:0.83840
    [38]	validation_0-auc:0.88961	validation_1-auc:0.83815
    [39]	validation_0-auc:0.88980	validation_1-auc:0.83797
    [0]	validation_0-auc:0.82297	validation_1-auc:0.81707
    [1]	validation_0-auc:0.83141	validation_1-auc:0.82133
    [2]	validation_0-auc:0.83805	validation_1-auc:0.82785
    [3]	validation_0-auc:0.84271	validation_1-auc:0.82901
    [4]	validation_0-auc:0.84766	validation_1-auc:0.83398
    [5]	validation_0-auc:0.85126	validation_1-auc:0.83381
    [6]	validation_0-auc:0.85486	validation_1-auc:0.83360
    [7]	validation_0-auc:0.85943	validation_1-auc:0.83287
    [8]	validation_0-auc:0.86243	validation_1-auc:0.83187
    [9]	validation_0-auc:0.86501	validation_1-auc:0.83304
    [10]	validation_0-auc:0.86496	validation_1-auc:0.83318
    [11]	validation_0-auc:0.86466	validation_1-auc:0.83180
    [12]	validation_0-auc:0.87013	validation_1-auc:0.83453
    [13]	validation_0-auc:0.87319	validation_1-auc:0.83548
    [14]	validation_0-auc:0.87479	validation_1-auc:0.83833
    [15]	validation_0-auc:0.87618	validation_1-auc:0.83862
    [16]	validation_0-auc:0.87739	validation_1-auc:0.83927
    [17]	validation_0-auc:0.87883	validation_1-auc:0.83904
    [18]	validation_0-auc:0.88009	validation_1-auc:0.83809
    [19]	validation_0-auc:0.88083	validation_1-auc:0.83801
    [20]	validation_0-auc:0.88192	validation_1-auc:0.83779
    [21]	validation_0-auc:0.88252	validation_1-auc:0.83713
    [22]	validation_0-auc:0.88381	validation_1-auc:0.83671
    [23]	validation_0-auc:0.88442	validation_1-auc:0.83659
    [24]	validation_0-auc:0.88472	validation_1-auc:0.83674
    [25]	validation_0-auc:0.88543	validation_1-auc:0.83674
    [26]	validation_0-auc:0.88568	validation_1-auc:0.83659
    [27]	validation_0-auc:0.88629	validation_1-auc:0.83675
    [28]	validation_0-auc:0.88698	validation_1-auc:0.83691
    [29]	validation_0-auc:0.88739	validation_1-auc:0.83677
    [30]	validation_0-auc:0.88752	validation_1-auc:0.83681
    [31]	validation_0-auc:0.88765	validation_1-auc:0.83666
    [32]	validation_0-auc:0.88796	validation_1-auc:0.83611
    [33]	validation_0-auc:0.88928	validation_1-auc:0.83562
    [34]	validation_0-auc:0.89022	validation_1-auc:0.83498
    [35]	validation_0-auc:0.89040	validation_1-auc:0.83515
    [36]	validation_0-auc:0.89105	validation_1-auc:0.83420
    [37]	validation_0-auc:0.89119	validation_1-auc:0.83435
    [38]	validation_0-auc:0.89161	validation_1-auc:0.83405
    [39]	validation_0-auc:0.89254	validation_1-auc:0.83376
    [40]	validation_0-auc:0.89268	validation_1-auc:0.83387
    [41]	validation_0-auc:0.89275	validation_1-auc:0.83334
    [42]	validation_0-auc:0.89324	validation_1-auc:0.83343
    [43]	validation_0-auc:0.89320	validation_1-auc:0.83370
    [44]	validation_0-auc:0.89401	validation_1-auc:0.83323
    [45]	validation_0-auc:0.89457	validation_1-auc:0.83295
    [0]	validation_0-auc:0.80839	validation_1-auc:0.80987
    [1]	validation_0-auc:0.82568	validation_1-auc:0.82196
    [2]	validation_0-auc:0.83339	validation_1-auc:0.82958
    [3]	validation_0-auc:0.83699	validation_1-auc:0.83249
    [4]	validation_0-auc:0.84250	validation_1-auc:0.83515
    [5]	validation_0-auc:0.84330	validation_1-auc:0.83849
    [6]	validation_0-auc:0.84276	validation_1-auc:0.83417
    [7]	validation_0-auc:0.84840	validation_1-auc:0.83701
    [8]	validation_0-auc:0.85092	validation_1-auc:0.83860
    [9]	validation_0-auc:0.85093	validation_1-auc:0.83816
    [10]	validation_0-auc:0.84973	validation_1-auc:0.83676
    [11]	validation_0-auc:0.85001	validation_1-auc:0.83660
    [12]	validation_0-auc:0.85677	validation_1-auc:0.84120
    [13]	validation_0-auc:0.86055	validation_1-auc:0.84387
    [14]	validation_0-auc:0.86264	validation_1-auc:0.84445
    [15]	validation_0-auc:0.86474	validation_1-auc:0.84328
    [16]	validation_0-auc:0.86582	validation_1-auc:0.84307
    [17]	validation_0-auc:0.86725	validation_1-auc:0.84240
    [18]	validation_0-auc:0.86782	validation_1-auc:0.84237
    [19]	validation_0-auc:0.86878	validation_1-auc:0.84274
    [20]	validation_0-auc:0.86929	validation_1-auc:0.84272
    [21]	validation_0-auc:0.86988	validation_1-auc:0.84276
    [22]	validation_0-auc:0.87099	validation_1-auc:0.84274
    [23]	validation_0-auc:0.87183	validation_1-auc:0.84258
    [24]	validation_0-auc:0.87287	validation_1-auc:0.84256
    [25]	validation_0-auc:0.87385	validation_1-auc:0.84239
    [26]	validation_0-auc:0.87472	validation_1-auc:0.84231
    [27]	validation_0-auc:0.87540	validation_1-auc:0.84245
    [28]	validation_0-auc:0.87601	validation_1-auc:0.84225
    [29]	validation_0-auc:0.87614	validation_1-auc:0.84249
    [30]	validation_0-auc:0.87669	validation_1-auc:0.84269
    [31]	validation_0-auc:0.87709	validation_1-auc:0.84240
    [32]	validation_0-auc:0.87732	validation_1-auc:0.84280
    [33]	validation_0-auc:0.87776	validation_1-auc:0.84267
    [34]	validation_0-auc:0.87798	validation_1-auc:0.84254
    [35]	validation_0-auc:0.87860	validation_1-auc:0.84245
    [36]	validation_0-auc:0.87929	validation_1-auc:0.84217
    [37]	validation_0-auc:0.87959	validation_1-auc:0.84184
    [38]	validation_0-auc:0.87981	validation_1-auc:0.84149
    [39]	validation_0-auc:0.88077	validation_1-auc:0.84133
    [40]	validation_0-auc:0.88102	validation_1-auc:0.84151
    [41]	validation_0-auc:0.88126	validation_1-auc:0.84146
    [42]	validation_0-auc:0.88221	validation_1-auc:0.84156
    [43]	validation_0-auc:0.88251	validation_1-auc:0.84143
    [44]	validation_0-auc:0.88287	validation_1-auc:0.84174
    GridSearchCV 최적 파라미머: {'colsample_bytree': 0.5, 'max_depth': 5, 'min_child_weight': 3}
    ROC AUC: 0.8445
    

## LightGBM 모델 학습과 하이퍼 파라미터 튜닝


```python
from lightgbm import LGBMClassifier

lgbm_clf=LGBMClassifier(n_estimators=500)

evals=[(X_test, y_test)]
lgbm_clf.fit(X_train, y_train, early_stopping_rounds=100, eval_metric="auc", eval_set=evals, verbose=True)
lgbm_roc_score=roc_auc_score(y_test, lgbm_clf.predict_proba(X_test)[:,1], average='macro')

print("ROC AUC: {0:.4f}".format(lgbm_roc_score))
```

    [1]	valid_0's auc: 0.817384	valid_0's binary_logloss: 0.165046
    Training until validation scores don't improve for 100 rounds
    [2]	valid_0's auc: 0.818903	valid_0's binary_logloss: 0.160006
    [3]	valid_0's auc: 0.827707	valid_0's binary_logloss: 0.156323
    [4]	valid_0's auc: 0.832155	valid_0's binary_logloss: 0.153463
    [5]	valid_0's auc: 0.834677	valid_0's binary_logloss: 0.151256
    [6]	valid_0's auc: 0.834093	valid_0's binary_logloss: 0.149427
    [7]	valid_0's auc: 0.837046	valid_0's binary_logloss: 0.147961
    [8]	valid_0's auc: 0.837838	valid_0's binary_logloss: 0.146591
    [9]	valid_0's auc: 0.839435	valid_0's binary_logloss: 0.145455
    [10]	valid_0's auc: 0.83973	valid_0's binary_logloss: 0.144486
    [11]	valid_0's auc: 0.839799	valid_0's binary_logloss: 0.143769
    [12]	valid_0's auc: 0.840034	valid_0's binary_logloss: 0.143146
    [13]	valid_0's auc: 0.840271	valid_0's binary_logloss: 0.142533
    [14]	valid_0's auc: 0.840342	valid_0's binary_logloss: 0.142036
    [15]	valid_0's auc: 0.840928	valid_0's binary_logloss: 0.14161
    [16]	valid_0's auc: 0.840337	valid_0's binary_logloss: 0.141307
    [17]	valid_0's auc: 0.839901	valid_0's binary_logloss: 0.141152
    [18]	valid_0's auc: 0.839742	valid_0's binary_logloss: 0.141018
    [19]	valid_0's auc: 0.839818	valid_0's binary_logloss: 0.14068
    [20]	valid_0's auc: 0.839307	valid_0's binary_logloss: 0.140562
    [21]	valid_0's auc: 0.839662	valid_0's binary_logloss: 0.140353
    [22]	valid_0's auc: 0.840411	valid_0's binary_logloss: 0.140144
    [23]	valid_0's auc: 0.840522	valid_0's binary_logloss: 0.139983
    [24]	valid_0's auc: 0.840208	valid_0's binary_logloss: 0.139943
    [25]	valid_0's auc: 0.839578	valid_0's binary_logloss: 0.139898
    [26]	valid_0's auc: 0.83975	valid_0's binary_logloss: 0.139814
    [27]	valid_0's auc: 0.83988	valid_0's binary_logloss: 0.139711
    [28]	valid_0's auc: 0.839704	valid_0's binary_logloss: 0.139681
    [29]	valid_0's auc: 0.839432	valid_0's binary_logloss: 0.139662
    [30]	valid_0's auc: 0.839196	valid_0's binary_logloss: 0.139641
    [31]	valid_0's auc: 0.838891	valid_0's binary_logloss: 0.139654
    [32]	valid_0's auc: 0.838943	valid_0's binary_logloss: 0.1396
    [33]	valid_0's auc: 0.838632	valid_0's binary_logloss: 0.139642
    [34]	valid_0's auc: 0.838314	valid_0's binary_logloss: 0.139687
    [35]	valid_0's auc: 0.83844	valid_0's binary_logloss: 0.139668
    [36]	valid_0's auc: 0.839074	valid_0's binary_logloss: 0.139562
    [37]	valid_0's auc: 0.838806	valid_0's binary_logloss: 0.139594
    [38]	valid_0's auc: 0.839041	valid_0's binary_logloss: 0.139574
    [39]	valid_0's auc: 0.839081	valid_0's binary_logloss: 0.139587
    [40]	valid_0's auc: 0.839276	valid_0's binary_logloss: 0.139504
    [41]	valid_0's auc: 0.83951	valid_0's binary_logloss: 0.139481
    [42]	valid_0's auc: 0.839544	valid_0's binary_logloss: 0.139487
    [43]	valid_0's auc: 0.839673	valid_0's binary_logloss: 0.139478
    [44]	valid_0's auc: 0.839677	valid_0's binary_logloss: 0.139453
    [45]	valid_0's auc: 0.839703	valid_0's binary_logloss: 0.139445
    [46]	valid_0's auc: 0.839601	valid_0's binary_logloss: 0.139468
    [47]	valid_0's auc: 0.839318	valid_0's binary_logloss: 0.139529
    [48]	valid_0's auc: 0.839462	valid_0's binary_logloss: 0.139486
    [49]	valid_0's auc: 0.839288	valid_0's binary_logloss: 0.139492
    [50]	valid_0's auc: 0.838987	valid_0's binary_logloss: 0.139572
    [51]	valid_0's auc: 0.838845	valid_0's binary_logloss: 0.139603
    [52]	valid_0's auc: 0.838655	valid_0's binary_logloss: 0.139623
    [53]	valid_0's auc: 0.838783	valid_0's binary_logloss: 0.139609
    [54]	valid_0's auc: 0.838695	valid_0's binary_logloss: 0.139638
    [55]	valid_0's auc: 0.838868	valid_0's binary_logloss: 0.139625
    [56]	valid_0's auc: 0.838653	valid_0's binary_logloss: 0.139645
    [57]	valid_0's auc: 0.83856	valid_0's binary_logloss: 0.139688
    [58]	valid_0's auc: 0.838475	valid_0's binary_logloss: 0.139694
    [59]	valid_0's auc: 0.8384	valid_0's binary_logloss: 0.139682
    [60]	valid_0's auc: 0.838319	valid_0's binary_logloss: 0.13969
    [61]	valid_0's auc: 0.838209	valid_0's binary_logloss: 0.13973
    [62]	valid_0's auc: 0.83806	valid_0's binary_logloss: 0.139765
    [63]	valid_0's auc: 0.838096	valid_0's binary_logloss: 0.139749
    [64]	valid_0's auc: 0.838163	valid_0's binary_logloss: 0.139746
    [65]	valid_0's auc: 0.838183	valid_0's binary_logloss: 0.139805
    [66]	valid_0's auc: 0.838215	valid_0's binary_logloss: 0.139815
    [67]	valid_0's auc: 0.838268	valid_0's binary_logloss: 0.139822
    [68]	valid_0's auc: 0.83836	valid_0's binary_logloss: 0.139816
    [69]	valid_0's auc: 0.838114	valid_0's binary_logloss: 0.139874
    [70]	valid_0's auc: 0.83832	valid_0's binary_logloss: 0.139816
    [71]	valid_0's auc: 0.838256	valid_0's binary_logloss: 0.139818
    [72]	valid_0's auc: 0.838231	valid_0's binary_logloss: 0.139845
    [73]	valid_0's auc: 0.838028	valid_0's binary_logloss: 0.139888
    [74]	valid_0's auc: 0.837912	valid_0's binary_logloss: 0.139905
    [75]	valid_0's auc: 0.83772	valid_0's binary_logloss: 0.13992
    [76]	valid_0's auc: 0.837606	valid_0's binary_logloss: 0.139899
    [77]	valid_0's auc: 0.837521	valid_0's binary_logloss: 0.139925
    [78]	valid_0's auc: 0.837462	valid_0's binary_logloss: 0.139957
    [79]	valid_0's auc: 0.837541	valid_0's binary_logloss: 0.139944
    [80]	valid_0's auc: 0.838013	valid_0's binary_logloss: 0.13983
    [81]	valid_0's auc: 0.83789	valid_0's binary_logloss: 0.139874
    [82]	valid_0's auc: 0.837671	valid_0's binary_logloss: 0.139975
    [83]	valid_0's auc: 0.837707	valid_0's binary_logloss: 0.139972
    [84]	valid_0's auc: 0.837631	valid_0's binary_logloss: 0.140011
    [85]	valid_0's auc: 0.837496	valid_0's binary_logloss: 0.140023
    [86]	valid_0's auc: 0.83757	valid_0's binary_logloss: 0.140021
    [87]	valid_0's auc: 0.837284	valid_0's binary_logloss: 0.140099
    [88]	valid_0's auc: 0.837228	valid_0's binary_logloss: 0.140115
    [89]	valid_0's auc: 0.836964	valid_0's binary_logloss: 0.140172
    [90]	valid_0's auc: 0.836752	valid_0's binary_logloss: 0.140225
    [91]	valid_0's auc: 0.836833	valid_0's binary_logloss: 0.140221
    [92]	valid_0's auc: 0.836648	valid_0's binary_logloss: 0.140277
    [93]	valid_0's auc: 0.836648	valid_0's binary_logloss: 0.140315
    [94]	valid_0's auc: 0.836677	valid_0's binary_logloss: 0.140321
    [95]	valid_0's auc: 0.836729	valid_0's binary_logloss: 0.140307
    [96]	valid_0's auc: 0.8368	valid_0's binary_logloss: 0.140313
    [97]	valid_0's auc: 0.836797	valid_0's binary_logloss: 0.140331
    [98]	valid_0's auc: 0.836675	valid_0's binary_logloss: 0.140361
    [99]	valid_0's auc: 0.83655	valid_0's binary_logloss: 0.14039
    [100]	valid_0's auc: 0.836518	valid_0's binary_logloss: 0.1404
    [101]	valid_0's auc: 0.836998	valid_0's binary_logloss: 0.140294
    [102]	valid_0's auc: 0.836778	valid_0's binary_logloss: 0.140366
    [103]	valid_0's auc: 0.83694	valid_0's binary_logloss: 0.140333
    [104]	valid_0's auc: 0.836749	valid_0's binary_logloss: 0.14039
    [105]	valid_0's auc: 0.836752	valid_0's binary_logloss: 0.140391
    [106]	valid_0's auc: 0.837197	valid_0's binary_logloss: 0.140305
    [107]	valid_0's auc: 0.837141	valid_0's binary_logloss: 0.140329
    [108]	valid_0's auc: 0.8371	valid_0's binary_logloss: 0.140344
    [109]	valid_0's auc: 0.837136	valid_0's binary_logloss: 0.14033
    [110]	valid_0's auc: 0.837102	valid_0's binary_logloss: 0.140388
    [111]	valid_0's auc: 0.836957	valid_0's binary_logloss: 0.140426
    [112]	valid_0's auc: 0.836779	valid_0's binary_logloss: 0.14051
    [113]	valid_0's auc: 0.836831	valid_0's binary_logloss: 0.140526
    [114]	valid_0's auc: 0.836783	valid_0's binary_logloss: 0.14055
    [115]	valid_0's auc: 0.836672	valid_0's binary_logloss: 0.140585
    Early stopping, best iteration is:
    [15]	valid_0's auc: 0.840928	valid_0's binary_logloss: 0.14161
    ROC AUC: 0.8409
    


```python
from sklearn.model_selection import GridSearchCV

lgbm_clf=LGBMClassifier(n_estimators=200)
params={'num_leaves':[32,64],
        'max_depth':[128, 160],
        'min_child_samples':[60,100],
        'subsample':[0.8,1]}

gridcv=GridSearchCV(lgbm_clf, param_grid=params, cv=3)
gridcv.fit(X_train, y_train, early_stopping_rounds=30, eval_metric="auc", eval_set=[(X_train, y_train),(X_test, y_test)])

print("GridSearchCV 최적 파라미머:",gridcv.best_params_)

lgbm_roc_score=roc_auc_score(y_test, gridcv.predict_proba(X_test)[:,1], average='macro')

print("ROC AUC: {0:.4f}".format(lgbm_roc_score))
```

    [1]	valid_0's auc: 0.820235	valid_0's binary_logloss: 0.156085	valid_1's auc: 0.81613	valid_1's binary_logloss: 0.164992
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.825778	valid_0's binary_logloss: 0.150951	valid_1's auc: 0.821835	valid_1's binary_logloss: 0.159874
    [3]	valid_0's auc: 0.832262	valid_0's binary_logloss: 0.147158	valid_1's auc: 0.826533	valid_1's binary_logloss: 0.156346
    [4]	valid_0's auc: 0.83865	valid_0's binary_logloss: 0.144126	valid_1's auc: 0.833166	valid_1's binary_logloss: 0.1534
    [5]	valid_0's auc: 0.842822	valid_0's binary_logloss: 0.141725	valid_1's auc: 0.836448	valid_1's binary_logloss: 0.151167
    [6]	valid_0's auc: 0.844702	valid_0's binary_logloss: 0.139642	valid_1's auc: 0.837094	valid_1's binary_logloss: 0.149356
    [7]	valid_0's auc: 0.847144	valid_0's binary_logloss: 0.13794	valid_1's auc: 0.837965	valid_1's binary_logloss: 0.147853
    [8]	valid_0's auc: 0.848277	valid_0's binary_logloss: 0.136499	valid_1's auc: 0.837663	valid_1's binary_logloss: 0.146543
    [9]	valid_0's auc: 0.849328	valid_0's binary_logloss: 0.135326	valid_1's auc: 0.837413	valid_1's binary_logloss: 0.145528
    [10]	valid_0's auc: 0.851112	valid_0's binary_logloss: 0.134188	valid_1's auc: 0.836954	valid_1's binary_logloss: 0.14466
    [11]	valid_0's auc: 0.852613	valid_0's binary_logloss: 0.133257	valid_1's auc: 0.837393	valid_1's binary_logloss: 0.143843
    [12]	valid_0's auc: 0.854906	valid_0's binary_logloss: 0.132346	valid_1's auc: 0.837459	valid_1's binary_logloss: 0.143285
    [13]	valid_0's auc: 0.855656	valid_0's binary_logloss: 0.131601	valid_1's auc: 0.837612	valid_1's binary_logloss: 0.142732
    [14]	valid_0's auc: 0.857076	valid_0's binary_logloss: 0.130884	valid_1's auc: 0.837055	valid_1's binary_logloss: 0.142403
    [15]	valid_0's auc: 0.857961	valid_0's binary_logloss: 0.130252	valid_1's auc: 0.837198	valid_1's binary_logloss: 0.142031
    [16]	valid_0's auc: 0.860191	valid_0's binary_logloss: 0.129596	valid_1's auc: 0.836016	valid_1's binary_logloss: 0.141822
    [17]	valid_0's auc: 0.860941	valid_0's binary_logloss: 0.129064	valid_1's auc: 0.836076	valid_1's binary_logloss: 0.141551
    [18]	valid_0's auc: 0.862201	valid_0's binary_logloss: 0.128565	valid_1's auc: 0.835929	valid_1's binary_logloss: 0.141326
    [19]	valid_0's auc: 0.863581	valid_0's binary_logloss: 0.128105	valid_1's auc: 0.835256	valid_1's binary_logloss: 0.141243
    [20]	valid_0's auc: 0.864799	valid_0's binary_logloss: 0.127654	valid_1's auc: 0.83435	valid_1's binary_logloss: 0.141148
    [21]	valid_0's auc: 0.866472	valid_0's binary_logloss: 0.127165	valid_1's auc: 0.834176	valid_1's binary_logloss: 0.141041
    [22]	valid_0's auc: 0.867055	valid_0's binary_logloss: 0.126777	valid_1's auc: 0.834173	valid_1's binary_logloss: 0.140887
    [23]	valid_0's auc: 0.867726	valid_0's binary_logloss: 0.12643	valid_1's auc: 0.833577	valid_1's binary_logloss: 0.140909
    [24]	valid_0's auc: 0.868612	valid_0's binary_logloss: 0.126061	valid_1's auc: 0.833336	valid_1's binary_logloss: 0.140824
    [25]	valid_0's auc: 0.869224	valid_0's binary_logloss: 0.125753	valid_1's auc: 0.833428	valid_1's binary_logloss: 0.140793
    [26]	valid_0's auc: 0.870183	valid_0's binary_logloss: 0.125414	valid_1's auc: 0.83333	valid_1's binary_logloss: 0.140724
    [27]	valid_0's auc: 0.870926	valid_0's binary_logloss: 0.125123	valid_1's auc: 0.832503	valid_1's binary_logloss: 0.140772
    [28]	valid_0's auc: 0.872431	valid_0's binary_logloss: 0.124766	valid_1's auc: 0.832826	valid_1's binary_logloss: 0.140685
    [29]	valid_0's auc: 0.873397	valid_0's binary_logloss: 0.124495	valid_1's auc: 0.833175	valid_1's binary_logloss: 0.140604
    [30]	valid_0's auc: 0.87475	valid_0's binary_logloss: 0.12417	valid_1's auc: 0.833614	valid_1's binary_logloss: 0.140497
    [31]	valid_0's auc: 0.875407	valid_0's binary_logloss: 0.12389	valid_1's auc: 0.833706	valid_1's binary_logloss: 0.140428
    [32]	valid_0's auc: 0.876136	valid_0's binary_logloss: 0.123637	valid_1's auc: 0.833458	valid_1's binary_logloss: 0.140448
    [33]	valid_0's auc: 0.876575	valid_0's binary_logloss: 0.123421	valid_1's auc: 0.832965	valid_1's binary_logloss: 0.140498
    [34]	valid_0's auc: 0.877224	valid_0's binary_logloss: 0.123219	valid_1's auc: 0.832659	valid_1's binary_logloss: 0.140537
    [35]	valid_0's auc: 0.877898	valid_0's binary_logloss: 0.122947	valid_1's auc: 0.832787	valid_1's binary_logloss: 0.140536
    [36]	valid_0's auc: 0.878334	valid_0's binary_logloss: 0.122724	valid_1's auc: 0.832724	valid_1's binary_logloss: 0.14053
    [37]	valid_0's auc: 0.878762	valid_0's binary_logloss: 0.122514	valid_1's auc: 0.832581	valid_1's binary_logloss: 0.140533
    Early stopping, best iteration is:
    [7]	valid_0's auc: 0.847144	valid_0's binary_logloss: 0.13794	valid_1's auc: 0.837965	valid_1's binary_logloss: 0.147853
    [1]	valid_0's auc: 0.814371	valid_0's binary_logloss: 0.156452	valid_1's auc: 0.813175	valid_1's binary_logloss: 0.165418
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.827277	valid_0's binary_logloss: 0.151084	valid_1's auc: 0.819635	valid_1's binary_logloss: 0.160159
    [3]	valid_0's auc: 0.837033	valid_0's binary_logloss: 0.14722	valid_1's auc: 0.828221	valid_1's binary_logloss: 0.156492
    [4]	valid_0's auc: 0.840167	valid_0's binary_logloss: 0.14423	valid_1's auc: 0.830942	valid_1's binary_logloss: 0.153586
    [5]	valid_0's auc: 0.842499	valid_0's binary_logloss: 0.141721	valid_1's auc: 0.833301	valid_1's binary_logloss: 0.151219
    [6]	valid_0's auc: 0.845403	valid_0's binary_logloss: 0.139708	valid_1's auc: 0.836412	valid_1's binary_logloss: 0.149312
    [7]	valid_0's auc: 0.848049	valid_0's binary_logloss: 0.138024	valid_1's auc: 0.836054	valid_1's binary_logloss: 0.14779
    [8]	valid_0's auc: 0.849694	valid_0's binary_logloss: 0.136542	valid_1's auc: 0.837537	valid_1's binary_logloss: 0.146417
    [9]	valid_0's auc: 0.851646	valid_0's binary_logloss: 0.135289	valid_1's auc: 0.838418	valid_1's binary_logloss: 0.145329
    [10]	valid_0's auc: 0.853642	valid_0's binary_logloss: 0.134189	valid_1's auc: 0.839342	valid_1's binary_logloss: 0.144374
    [11]	valid_0's auc: 0.855647	valid_0's binary_logloss: 0.133227	valid_1's auc: 0.840035	valid_1's binary_logloss: 0.143552
    [12]	valid_0's auc: 0.856768	valid_0's binary_logloss: 0.132399	valid_1's auc: 0.839294	valid_1's binary_logloss: 0.143047
    [13]	valid_0's auc: 0.85763	valid_0's binary_logloss: 0.13165	valid_1's auc: 0.838911	valid_1's binary_logloss: 0.142469
    [14]	valid_0's auc: 0.859243	valid_0's binary_logloss: 0.130936	valid_1's auc: 0.838705	valid_1's binary_logloss: 0.141913
    [15]	valid_0's auc: 0.860124	valid_0's binary_logloss: 0.130312	valid_1's auc: 0.838608	valid_1's binary_logloss: 0.141547
    [16]	valid_0's auc: 0.861358	valid_0's binary_logloss: 0.129687	valid_1's auc: 0.838422	valid_1's binary_logloss: 0.141134
    [17]	valid_0's auc: 0.862159	valid_0's binary_logloss: 0.129139	valid_1's auc: 0.838636	valid_1's binary_logloss: 0.140786
    [18]	valid_0's auc: 0.862729	valid_0's binary_logloss: 0.128664	valid_1's auc: 0.838342	valid_1's binary_logloss: 0.140538
    [19]	valid_0's auc: 0.863842	valid_0's binary_logloss: 0.128137	valid_1's auc: 0.838464	valid_1's binary_logloss: 0.140331
    [20]	valid_0's auc: 0.864859	valid_0's binary_logloss: 0.127657	valid_1's auc: 0.837832	valid_1's binary_logloss: 0.140179
    [21]	valid_0's auc: 0.866227	valid_0's binary_logloss: 0.127137	valid_1's auc: 0.837735	valid_1's binary_logloss: 0.140043
    [22]	valid_0's auc: 0.866925	valid_0's binary_logloss: 0.126772	valid_1's auc: 0.838268	valid_1's binary_logloss: 0.139927
    [23]	valid_0's auc: 0.867727	valid_0's binary_logloss: 0.126369	valid_1's auc: 0.838482	valid_1's binary_logloss: 0.139787
    [24]	valid_0's auc: 0.868239	valid_0's binary_logloss: 0.126013	valid_1's auc: 0.838767	valid_1's binary_logloss: 0.13964
    [25]	valid_0's auc: 0.869304	valid_0's binary_logloss: 0.125622	valid_1's auc: 0.838562	valid_1's binary_logloss: 0.139648
    [26]	valid_0's auc: 0.870347	valid_0's binary_logloss: 0.125288	valid_1's auc: 0.838228	valid_1's binary_logloss: 0.139618
    [27]	valid_0's auc: 0.871198	valid_0's binary_logloss: 0.124953	valid_1's auc: 0.838403	valid_1's binary_logloss: 0.139594
    [28]	valid_0's auc: 0.872024	valid_0's binary_logloss: 0.124672	valid_1's auc: 0.838405	valid_1's binary_logloss: 0.139526
    [29]	valid_0's auc: 0.873184	valid_0's binary_logloss: 0.124303	valid_1's auc: 0.838211	valid_1's binary_logloss: 0.139531
    [30]	valid_0's auc: 0.874076	valid_0's binary_logloss: 0.12403	valid_1's auc: 0.838983	valid_1's binary_logloss: 0.139411
    [31]	valid_0's auc: 0.874768	valid_0's binary_logloss: 0.123745	valid_1's auc: 0.839314	valid_1's binary_logloss: 0.139314
    [32]	valid_0's auc: 0.875593	valid_0's binary_logloss: 0.123486	valid_1's auc: 0.838875	valid_1's binary_logloss: 0.139322
    [33]	valid_0's auc: 0.8767	valid_0's binary_logloss: 0.123182	valid_1's auc: 0.838809	valid_1's binary_logloss: 0.139329
    [34]	valid_0's auc: 0.87774	valid_0's binary_logloss: 0.122892	valid_1's auc: 0.838376	valid_1's binary_logloss: 0.139342
    [35]	valid_0's auc: 0.878372	valid_0's binary_logloss: 0.122634	valid_1's auc: 0.838454	valid_1's binary_logloss: 0.13931
    [36]	valid_0's auc: 0.879098	valid_0's binary_logloss: 0.122414	valid_1's auc: 0.838895	valid_1's binary_logloss: 0.13925
    [37]	valid_0's auc: 0.879502	valid_0's binary_logloss: 0.122216	valid_1's auc: 0.838441	valid_1's binary_logloss: 0.139302
    [38]	valid_0's auc: 0.880036	valid_0's binary_logloss: 0.121998	valid_1's auc: 0.838582	valid_1's binary_logloss: 0.139306
    [39]	valid_0's auc: 0.880641	valid_0's binary_logloss: 0.121716	valid_1's auc: 0.838787	valid_1's binary_logloss: 0.139269
    [40]	valid_0's auc: 0.881249	valid_0's binary_logloss: 0.121482	valid_1's auc: 0.838906	valid_1's binary_logloss: 0.139223
    [41]	valid_0's auc: 0.881919	valid_0's binary_logloss: 0.121223	valid_1's auc: 0.838567	valid_1's binary_logloss: 0.13926
    Early stopping, best iteration is:
    [11]	valid_0's auc: 0.855647	valid_0's binary_logloss: 0.133227	valid_1's auc: 0.840035	valid_1's binary_logloss: 0.143552
    [1]	valid_0's auc: 0.821645	valid_0's binary_logloss: 0.156528	valid_1's auc: 0.81857	valid_1's binary_logloss: 0.165101
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.827488	valid_0's binary_logloss: 0.151189	valid_1's auc: 0.822299	valid_1's binary_logloss: 0.160072
    [3]	valid_0's auc: 0.837855	valid_0's binary_logloss: 0.147263	valid_1's auc: 0.829855	valid_1's binary_logloss: 0.156527
    [4]	valid_0's auc: 0.840063	valid_0's binary_logloss: 0.144261	valid_1's auc: 0.833088	valid_1's binary_logloss: 0.153446
    [5]	valid_0's auc: 0.842802	valid_0's binary_logloss: 0.141691	valid_1's auc: 0.834541	valid_1's binary_logloss: 0.151144
    [6]	valid_0's auc: 0.844	valid_0's binary_logloss: 0.139654	valid_1's auc: 0.834542	valid_1's binary_logloss: 0.149333
    [7]	valid_0's auc: 0.845838	valid_0's binary_logloss: 0.138002	valid_1's auc: 0.835645	valid_1's binary_logloss: 0.147676
    [8]	valid_0's auc: 0.846869	valid_0's binary_logloss: 0.136628	valid_1's auc: 0.836118	valid_1's binary_logloss: 0.146491
    [9]	valid_0's auc: 0.849282	valid_0's binary_logloss: 0.135382	valid_1's auc: 0.837542	valid_1's binary_logloss: 0.14539
    [10]	valid_0's auc: 0.851021	valid_0's binary_logloss: 0.134282	valid_1's auc: 0.836942	valid_1's binary_logloss: 0.144584
    [11]	valid_0's auc: 0.852037	valid_0's binary_logloss: 0.133358	valid_1's auc: 0.8374	valid_1's binary_logloss: 0.143836
    [12]	valid_0's auc: 0.854496	valid_0's binary_logloss: 0.132505	valid_1's auc: 0.838593	valid_1's binary_logloss: 0.143171
    [13]	valid_0's auc: 0.857514	valid_0's binary_logloss: 0.131695	valid_1's auc: 0.838558	valid_1's binary_logloss: 0.142646
    [14]	valid_0's auc: 0.858827	valid_0's binary_logloss: 0.131006	valid_1's auc: 0.838498	valid_1's binary_logloss: 0.142158
    [15]	valid_0's auc: 0.860574	valid_0's binary_logloss: 0.130352	valid_1's auc: 0.837435	valid_1's binary_logloss: 0.141868
    [16]	valid_0's auc: 0.861239	valid_0's binary_logloss: 0.129765	valid_1's auc: 0.837374	valid_1's binary_logloss: 0.141537
    [17]	valid_0's auc: 0.86217	valid_0's binary_logloss: 0.129164	valid_1's auc: 0.837703	valid_1's binary_logloss: 0.141192
    [18]	valid_0's auc: 0.863228	valid_0's binary_logloss: 0.128615	valid_1's auc: 0.837526	valid_1's binary_logloss: 0.140917
    [19]	valid_0's auc: 0.86473	valid_0's binary_logloss: 0.128113	valid_1's auc: 0.838235	valid_1's binary_logloss: 0.140572
    [20]	valid_0's auc: 0.865797	valid_0's binary_logloss: 0.127679	valid_1's auc: 0.838788	valid_1's binary_logloss: 0.140332
    [21]	valid_0's auc: 0.866561	valid_0's binary_logloss: 0.127235	valid_1's auc: 0.839171	valid_1's binary_logloss: 0.140108
    [22]	valid_0's auc: 0.867237	valid_0's binary_logloss: 0.12688	valid_1's auc: 0.839213	valid_1's binary_logloss: 0.13991
    [23]	valid_0's auc: 0.867894	valid_0's binary_logloss: 0.126519	valid_1's auc: 0.839641	valid_1's binary_logloss: 0.139745
    [24]	valid_0's auc: 0.868501	valid_0's binary_logloss: 0.126192	valid_1's auc: 0.840025	valid_1's binary_logloss: 0.139593
    [25]	valid_0's auc: 0.869311	valid_0's binary_logloss: 0.125838	valid_1's auc: 0.839961	valid_1's binary_logloss: 0.139531
    [26]	valid_0's auc: 0.870325	valid_0's binary_logloss: 0.125518	valid_1's auc: 0.839261	valid_1's binary_logloss: 0.139524
    [27]	valid_0's auc: 0.871488	valid_0's binary_logloss: 0.125147	valid_1's auc: 0.839671	valid_1's binary_logloss: 0.139365
    [28]	valid_0's auc: 0.87235	valid_0's binary_logloss: 0.12484	valid_1's auc: 0.840114	valid_1's binary_logloss: 0.139236
    [29]	valid_0's auc: 0.872991	valid_0's binary_logloss: 0.124593	valid_1's auc: 0.839491	valid_1's binary_logloss: 0.139271
    [30]	valid_0's auc: 0.874129	valid_0's binary_logloss: 0.124312	valid_1's auc: 0.839589	valid_1's binary_logloss: 0.13918
    [31]	valid_0's auc: 0.875305	valid_0's binary_logloss: 0.123988	valid_1's auc: 0.839441	valid_1's binary_logloss: 0.139184
    [32]	valid_0's auc: 0.875943	valid_0's binary_logloss: 0.123748	valid_1's auc: 0.839268	valid_1's binary_logloss: 0.13919
    [33]	valid_0's auc: 0.876575	valid_0's binary_logloss: 0.123484	valid_1's auc: 0.839549	valid_1's binary_logloss: 0.139075
    [34]	valid_0's auc: 0.877426	valid_0's binary_logloss: 0.123156	valid_1's auc: 0.839087	valid_1's binary_logloss: 0.139148
    [35]	valid_0's auc: 0.87822	valid_0's binary_logloss: 0.122873	valid_1's auc: 0.8389	valid_1's binary_logloss: 0.139187
    [36]	valid_0's auc: 0.878932	valid_0's binary_logloss: 0.12259	valid_1's auc: 0.838921	valid_1's binary_logloss: 0.139194
    [37]	valid_0's auc: 0.879842	valid_0's binary_logloss: 0.12233	valid_1's auc: 0.839133	valid_1's binary_logloss: 0.139161
    [38]	valid_0's auc: 0.880497	valid_0's binary_logloss: 0.12208	valid_1's auc: 0.838975	valid_1's binary_logloss: 0.139143
    [39]	valid_0's auc: 0.881056	valid_0's binary_logloss: 0.121827	valid_1's auc: 0.839037	valid_1's binary_logloss: 0.139138
    [40]	valid_0's auc: 0.881604	valid_0's binary_logloss: 0.121603	valid_1's auc: 0.839204	valid_1's binary_logloss: 0.139119
    [41]	valid_0's auc: 0.882159	valid_0's binary_logloss: 0.121355	valid_1's auc: 0.839277	valid_1's binary_logloss: 0.139091
    [42]	valid_0's auc: 0.882757	valid_0's binary_logloss: 0.121116	valid_1's auc: 0.838964	valid_1's binary_logloss: 0.139133
    [43]	valid_0's auc: 0.883143	valid_0's binary_logloss: 0.120918	valid_1's auc: 0.839024	valid_1's binary_logloss: 0.139124
    [44]	valid_0's auc: 0.883697	valid_0's binary_logloss: 0.12072	valid_1's auc: 0.838652	valid_1's binary_logloss: 0.139203
    [45]	valid_0's auc: 0.884292	valid_0's binary_logloss: 0.120482	valid_1's auc: 0.839016	valid_1's binary_logloss: 0.139124
    [46]	valid_0's auc: 0.884969	valid_0's binary_logloss: 0.120266	valid_1's auc: 0.838683	valid_1's binary_logloss: 0.139184
    [47]	valid_0's auc: 0.8853	valid_0's binary_logloss: 0.120089	valid_1's auc: 0.838624	valid_1's binary_logloss: 0.139193
    [48]	valid_0's auc: 0.885876	valid_0's binary_logloss: 0.11993	valid_1's auc: 0.838569	valid_1's binary_logloss: 0.139212
    [49]	valid_0's auc: 0.886141	valid_0's binary_logloss: 0.119757	valid_1's auc: 0.838345	valid_1's binary_logloss: 0.139288
    [50]	valid_0's auc: 0.886433	valid_0's binary_logloss: 0.119595	valid_1's auc: 0.838342	valid_1's binary_logloss: 0.139332
    [51]	valid_0's auc: 0.886975	valid_0's binary_logloss: 0.119377	valid_1's auc: 0.838335	valid_1's binary_logloss: 0.139331
    [52]	valid_0's auc: 0.887568	valid_0's binary_logloss: 0.119161	valid_1's auc: 0.838204	valid_1's binary_logloss: 0.139331
    [53]	valid_0's auc: 0.887867	valid_0's binary_logloss: 0.118974	valid_1's auc: 0.838044	valid_1's binary_logloss: 0.13936
    [54]	valid_0's auc: 0.888093	valid_0's binary_logloss: 0.118834	valid_1's auc: 0.838137	valid_1's binary_logloss: 0.13935
    [55]	valid_0's auc: 0.888289	valid_0's binary_logloss: 0.118675	valid_1's auc: 0.837878	valid_1's binary_logloss: 0.139392
    [56]	valid_0's auc: 0.888615	valid_0's binary_logloss: 0.118561	valid_1's auc: 0.837776	valid_1's binary_logloss: 0.139418
    [57]	valid_0's auc: 0.889157	valid_0's binary_logloss: 0.118369	valid_1's auc: 0.837585	valid_1's binary_logloss: 0.139447
    [58]	valid_0's auc: 0.889659	valid_0's binary_logloss: 0.11819	valid_1's auc: 0.837789	valid_1's binary_logloss: 0.139431
    Early stopping, best iteration is:
    [28]	valid_0's auc: 0.87235	valid_0's binary_logloss: 0.12484	valid_1's auc: 0.840114	valid_1's binary_logloss: 0.139236
    [1]	valid_0's auc: 0.820235	valid_0's binary_logloss: 0.156085	valid_1's auc: 0.81613	valid_1's binary_logloss: 0.164992
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.825778	valid_0's binary_logloss: 0.150951	valid_1's auc: 0.821835	valid_1's binary_logloss: 0.159874
    [3]	valid_0's auc: 0.832262	valid_0's binary_logloss: 0.147158	valid_1's auc: 0.826533	valid_1's binary_logloss: 0.156346
    [4]	valid_0's auc: 0.83865	valid_0's binary_logloss: 0.144126	valid_1's auc: 0.833166	valid_1's binary_logloss: 0.1534
    [5]	valid_0's auc: 0.842822	valid_0's binary_logloss: 0.141725	valid_1's auc: 0.836448	valid_1's binary_logloss: 0.151167
    [6]	valid_0's auc: 0.844702	valid_0's binary_logloss: 0.139642	valid_1's auc: 0.837094	valid_1's binary_logloss: 0.149356
    [7]	valid_0's auc: 0.847144	valid_0's binary_logloss: 0.13794	valid_1's auc: 0.837965	valid_1's binary_logloss: 0.147853
    [8]	valid_0's auc: 0.848277	valid_0's binary_logloss: 0.136499	valid_1's auc: 0.837663	valid_1's binary_logloss: 0.146543
    [9]	valid_0's auc: 0.849328	valid_0's binary_logloss: 0.135326	valid_1's auc: 0.837413	valid_1's binary_logloss: 0.145528
    [10]	valid_0's auc: 0.851112	valid_0's binary_logloss: 0.134188	valid_1's auc: 0.836954	valid_1's binary_logloss: 0.14466
    [11]	valid_0's auc: 0.852613	valid_0's binary_logloss: 0.133257	valid_1's auc: 0.837393	valid_1's binary_logloss: 0.143843
    [12]	valid_0's auc: 0.854906	valid_0's binary_logloss: 0.132346	valid_1's auc: 0.837459	valid_1's binary_logloss: 0.143285
    [13]	valid_0's auc: 0.855656	valid_0's binary_logloss: 0.131601	valid_1's auc: 0.837612	valid_1's binary_logloss: 0.142732
    [14]	valid_0's auc: 0.857076	valid_0's binary_logloss: 0.130884	valid_1's auc: 0.837055	valid_1's binary_logloss: 0.142403
    [15]	valid_0's auc: 0.857961	valid_0's binary_logloss: 0.130252	valid_1's auc: 0.837198	valid_1's binary_logloss: 0.142031
    [16]	valid_0's auc: 0.860191	valid_0's binary_logloss: 0.129596	valid_1's auc: 0.836016	valid_1's binary_logloss: 0.141822
    [17]	valid_0's auc: 0.860941	valid_0's binary_logloss: 0.129064	valid_1's auc: 0.836076	valid_1's binary_logloss: 0.141551
    [18]	valid_0's auc: 0.862201	valid_0's binary_logloss: 0.128565	valid_1's auc: 0.835929	valid_1's binary_logloss: 0.141326
    [19]	valid_0's auc: 0.863581	valid_0's binary_logloss: 0.128105	valid_1's auc: 0.835256	valid_1's binary_logloss: 0.141243
    [20]	valid_0's auc: 0.864799	valid_0's binary_logloss: 0.127654	valid_1's auc: 0.83435	valid_1's binary_logloss: 0.141148
    [21]	valid_0's auc: 0.866472	valid_0's binary_logloss: 0.127165	valid_1's auc: 0.834176	valid_1's binary_logloss: 0.141041
    [22]	valid_0's auc: 0.867055	valid_0's binary_logloss: 0.126777	valid_1's auc: 0.834173	valid_1's binary_logloss: 0.140887
    [23]	valid_0's auc: 0.867726	valid_0's binary_logloss: 0.12643	valid_1's auc: 0.833577	valid_1's binary_logloss: 0.140909
    [24]	valid_0's auc: 0.868612	valid_0's binary_logloss: 0.126061	valid_1's auc: 0.833336	valid_1's binary_logloss: 0.140824
    [25]	valid_0's auc: 0.869224	valid_0's binary_logloss: 0.125753	valid_1's auc: 0.833428	valid_1's binary_logloss: 0.140793
    [26]	valid_0's auc: 0.870183	valid_0's binary_logloss: 0.125414	valid_1's auc: 0.83333	valid_1's binary_logloss: 0.140724
    [27]	valid_0's auc: 0.870926	valid_0's binary_logloss: 0.125123	valid_1's auc: 0.832503	valid_1's binary_logloss: 0.140772
    [28]	valid_0's auc: 0.872431	valid_0's binary_logloss: 0.124766	valid_1's auc: 0.832826	valid_1's binary_logloss: 0.140685
    [29]	valid_0's auc: 0.873397	valid_0's binary_logloss: 0.124495	valid_1's auc: 0.833175	valid_1's binary_logloss: 0.140604
    [30]	valid_0's auc: 0.87475	valid_0's binary_logloss: 0.12417	valid_1's auc: 0.833614	valid_1's binary_logloss: 0.140497
    [31]	valid_0's auc: 0.875407	valid_0's binary_logloss: 0.12389	valid_1's auc: 0.833706	valid_1's binary_logloss: 0.140428
    [32]	valid_0's auc: 0.876136	valid_0's binary_logloss: 0.123637	valid_1's auc: 0.833458	valid_1's binary_logloss: 0.140448
    [33]	valid_0's auc: 0.876575	valid_0's binary_logloss: 0.123421	valid_1's auc: 0.832965	valid_1's binary_logloss: 0.140498
    [34]	valid_0's auc: 0.877224	valid_0's binary_logloss: 0.123219	valid_1's auc: 0.832659	valid_1's binary_logloss: 0.140537
    [35]	valid_0's auc: 0.877898	valid_0's binary_logloss: 0.122947	valid_1's auc: 0.832787	valid_1's binary_logloss: 0.140536
    [36]	valid_0's auc: 0.878334	valid_0's binary_logloss: 0.122724	valid_1's auc: 0.832724	valid_1's binary_logloss: 0.14053
    [37]	valid_0's auc: 0.878762	valid_0's binary_logloss: 0.122514	valid_1's auc: 0.832581	valid_1's binary_logloss: 0.140533
    Early stopping, best iteration is:
    [7]	valid_0's auc: 0.847144	valid_0's binary_logloss: 0.13794	valid_1's auc: 0.837965	valid_1's binary_logloss: 0.147853
    [1]	valid_0's auc: 0.814371	valid_0's binary_logloss: 0.156452	valid_1's auc: 0.813175	valid_1's binary_logloss: 0.165418
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.827277	valid_0's binary_logloss: 0.151084	valid_1's auc: 0.819635	valid_1's binary_logloss: 0.160159
    [3]	valid_0's auc: 0.837033	valid_0's binary_logloss: 0.14722	valid_1's auc: 0.828221	valid_1's binary_logloss: 0.156492
    [4]	valid_0's auc: 0.840167	valid_0's binary_logloss: 0.14423	valid_1's auc: 0.830942	valid_1's binary_logloss: 0.153586
    [5]	valid_0's auc: 0.842499	valid_0's binary_logloss: 0.141721	valid_1's auc: 0.833301	valid_1's binary_logloss: 0.151219
    [6]	valid_0's auc: 0.845403	valid_0's binary_logloss: 0.139708	valid_1's auc: 0.836412	valid_1's binary_logloss: 0.149312
    [7]	valid_0's auc: 0.848049	valid_0's binary_logloss: 0.138024	valid_1's auc: 0.836054	valid_1's binary_logloss: 0.14779
    [8]	valid_0's auc: 0.849694	valid_0's binary_logloss: 0.136542	valid_1's auc: 0.837537	valid_1's binary_logloss: 0.146417
    [9]	valid_0's auc: 0.851646	valid_0's binary_logloss: 0.135289	valid_1's auc: 0.838418	valid_1's binary_logloss: 0.145329
    [10]	valid_0's auc: 0.853642	valid_0's binary_logloss: 0.134189	valid_1's auc: 0.839342	valid_1's binary_logloss: 0.144374
    [11]	valid_0's auc: 0.855647	valid_0's binary_logloss: 0.133227	valid_1's auc: 0.840035	valid_1's binary_logloss: 0.143552
    [12]	valid_0's auc: 0.856768	valid_0's binary_logloss: 0.132399	valid_1's auc: 0.839294	valid_1's binary_logloss: 0.143047
    [13]	valid_0's auc: 0.85763	valid_0's binary_logloss: 0.13165	valid_1's auc: 0.838911	valid_1's binary_logloss: 0.142469
    [14]	valid_0's auc: 0.859243	valid_0's binary_logloss: 0.130936	valid_1's auc: 0.838705	valid_1's binary_logloss: 0.141913
    [15]	valid_0's auc: 0.860124	valid_0's binary_logloss: 0.130312	valid_1's auc: 0.838608	valid_1's binary_logloss: 0.141547
    [16]	valid_0's auc: 0.861358	valid_0's binary_logloss: 0.129687	valid_1's auc: 0.838422	valid_1's binary_logloss: 0.141134
    [17]	valid_0's auc: 0.862159	valid_0's binary_logloss: 0.129139	valid_1's auc: 0.838636	valid_1's binary_logloss: 0.140786
    [18]	valid_0's auc: 0.862729	valid_0's binary_logloss: 0.128664	valid_1's auc: 0.838342	valid_1's binary_logloss: 0.140538
    [19]	valid_0's auc: 0.863842	valid_0's binary_logloss: 0.128137	valid_1's auc: 0.838464	valid_1's binary_logloss: 0.140331
    [20]	valid_0's auc: 0.864859	valid_0's binary_logloss: 0.127657	valid_1's auc: 0.837832	valid_1's binary_logloss: 0.140179
    [21]	valid_0's auc: 0.866227	valid_0's binary_logloss: 0.127137	valid_1's auc: 0.837735	valid_1's binary_logloss: 0.140043
    [22]	valid_0's auc: 0.866925	valid_0's binary_logloss: 0.126772	valid_1's auc: 0.838268	valid_1's binary_logloss: 0.139927
    [23]	valid_0's auc: 0.867727	valid_0's binary_logloss: 0.126369	valid_1's auc: 0.838482	valid_1's binary_logloss: 0.139787
    [24]	valid_0's auc: 0.868239	valid_0's binary_logloss: 0.126013	valid_1's auc: 0.838767	valid_1's binary_logloss: 0.13964
    [25]	valid_0's auc: 0.869304	valid_0's binary_logloss: 0.125622	valid_1's auc: 0.838562	valid_1's binary_logloss: 0.139648
    [26]	valid_0's auc: 0.870347	valid_0's binary_logloss: 0.125288	valid_1's auc: 0.838228	valid_1's binary_logloss: 0.139618
    [27]	valid_0's auc: 0.871198	valid_0's binary_logloss: 0.124953	valid_1's auc: 0.838403	valid_1's binary_logloss: 0.139594
    [28]	valid_0's auc: 0.872024	valid_0's binary_logloss: 0.124672	valid_1's auc: 0.838405	valid_1's binary_logloss: 0.139526
    [29]	valid_0's auc: 0.873184	valid_0's binary_logloss: 0.124303	valid_1's auc: 0.838211	valid_1's binary_logloss: 0.139531
    [30]	valid_0's auc: 0.874076	valid_0's binary_logloss: 0.12403	valid_1's auc: 0.838983	valid_1's binary_logloss: 0.139411
    [31]	valid_0's auc: 0.874768	valid_0's binary_logloss: 0.123745	valid_1's auc: 0.839314	valid_1's binary_logloss: 0.139314
    [32]	valid_0's auc: 0.875593	valid_0's binary_logloss: 0.123486	valid_1's auc: 0.838875	valid_1's binary_logloss: 0.139322
    [33]	valid_0's auc: 0.8767	valid_0's binary_logloss: 0.123182	valid_1's auc: 0.838809	valid_1's binary_logloss: 0.139329
    [34]	valid_0's auc: 0.87774	valid_0's binary_logloss: 0.122892	valid_1's auc: 0.838376	valid_1's binary_logloss: 0.139342
    [35]	valid_0's auc: 0.878372	valid_0's binary_logloss: 0.122634	valid_1's auc: 0.838454	valid_1's binary_logloss: 0.13931
    [36]	valid_0's auc: 0.879098	valid_0's binary_logloss: 0.122414	valid_1's auc: 0.838895	valid_1's binary_logloss: 0.13925
    [37]	valid_0's auc: 0.879502	valid_0's binary_logloss: 0.122216	valid_1's auc: 0.838441	valid_1's binary_logloss: 0.139302
    [38]	valid_0's auc: 0.880036	valid_0's binary_logloss: 0.121998	valid_1's auc: 0.838582	valid_1's binary_logloss: 0.139306
    [39]	valid_0's auc: 0.880641	valid_0's binary_logloss: 0.121716	valid_1's auc: 0.838787	valid_1's binary_logloss: 0.139269
    [40]	valid_0's auc: 0.881249	valid_0's binary_logloss: 0.121482	valid_1's auc: 0.838906	valid_1's binary_logloss: 0.139223
    [41]	valid_0's auc: 0.881919	valid_0's binary_logloss: 0.121223	valid_1's auc: 0.838567	valid_1's binary_logloss: 0.13926
    Early stopping, best iteration is:
    [11]	valid_0's auc: 0.855647	valid_0's binary_logloss: 0.133227	valid_1's auc: 0.840035	valid_1's binary_logloss: 0.143552
    [1]	valid_0's auc: 0.821645	valid_0's binary_logloss: 0.156528	valid_1's auc: 0.81857	valid_1's binary_logloss: 0.165101
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.827488	valid_0's binary_logloss: 0.151189	valid_1's auc: 0.822299	valid_1's binary_logloss: 0.160072
    [3]	valid_0's auc: 0.837855	valid_0's binary_logloss: 0.147263	valid_1's auc: 0.829855	valid_1's binary_logloss: 0.156527
    [4]	valid_0's auc: 0.840063	valid_0's binary_logloss: 0.144261	valid_1's auc: 0.833088	valid_1's binary_logloss: 0.153446
    [5]	valid_0's auc: 0.842802	valid_0's binary_logloss: 0.141691	valid_1's auc: 0.834541	valid_1's binary_logloss: 0.151144
    [6]	valid_0's auc: 0.844	valid_0's binary_logloss: 0.139654	valid_1's auc: 0.834542	valid_1's binary_logloss: 0.149333
    [7]	valid_0's auc: 0.845838	valid_0's binary_logloss: 0.138002	valid_1's auc: 0.835645	valid_1's binary_logloss: 0.147676
    [8]	valid_0's auc: 0.846869	valid_0's binary_logloss: 0.136628	valid_1's auc: 0.836118	valid_1's binary_logloss: 0.146491
    [9]	valid_0's auc: 0.849282	valid_0's binary_logloss: 0.135382	valid_1's auc: 0.837542	valid_1's binary_logloss: 0.14539
    [10]	valid_0's auc: 0.851021	valid_0's binary_logloss: 0.134282	valid_1's auc: 0.836942	valid_1's binary_logloss: 0.144584
    [11]	valid_0's auc: 0.852037	valid_0's binary_logloss: 0.133358	valid_1's auc: 0.8374	valid_1's binary_logloss: 0.143836
    [12]	valid_0's auc: 0.854496	valid_0's binary_logloss: 0.132505	valid_1's auc: 0.838593	valid_1's binary_logloss: 0.143171
    [13]	valid_0's auc: 0.857514	valid_0's binary_logloss: 0.131695	valid_1's auc: 0.838558	valid_1's binary_logloss: 0.142646
    [14]	valid_0's auc: 0.858827	valid_0's binary_logloss: 0.131006	valid_1's auc: 0.838498	valid_1's binary_logloss: 0.142158
    [15]	valid_0's auc: 0.860574	valid_0's binary_logloss: 0.130352	valid_1's auc: 0.837435	valid_1's binary_logloss: 0.141868
    [16]	valid_0's auc: 0.861239	valid_0's binary_logloss: 0.129765	valid_1's auc: 0.837374	valid_1's binary_logloss: 0.141537
    [17]	valid_0's auc: 0.86217	valid_0's binary_logloss: 0.129164	valid_1's auc: 0.837703	valid_1's binary_logloss: 0.141192
    [18]	valid_0's auc: 0.863228	valid_0's binary_logloss: 0.128615	valid_1's auc: 0.837526	valid_1's binary_logloss: 0.140917
    [19]	valid_0's auc: 0.86473	valid_0's binary_logloss: 0.128113	valid_1's auc: 0.838235	valid_1's binary_logloss: 0.140572
    [20]	valid_0's auc: 0.865797	valid_0's binary_logloss: 0.127679	valid_1's auc: 0.838788	valid_1's binary_logloss: 0.140332
    [21]	valid_0's auc: 0.866561	valid_0's binary_logloss: 0.127235	valid_1's auc: 0.839171	valid_1's binary_logloss: 0.140108
    [22]	valid_0's auc: 0.867237	valid_0's binary_logloss: 0.12688	valid_1's auc: 0.839213	valid_1's binary_logloss: 0.13991
    [23]	valid_0's auc: 0.867894	valid_0's binary_logloss: 0.126519	valid_1's auc: 0.839641	valid_1's binary_logloss: 0.139745
    [24]	valid_0's auc: 0.868501	valid_0's binary_logloss: 0.126192	valid_1's auc: 0.840025	valid_1's binary_logloss: 0.139593
    [25]	valid_0's auc: 0.869311	valid_0's binary_logloss: 0.125838	valid_1's auc: 0.839961	valid_1's binary_logloss: 0.139531
    [26]	valid_0's auc: 0.870325	valid_0's binary_logloss: 0.125518	valid_1's auc: 0.839261	valid_1's binary_logloss: 0.139524
    [27]	valid_0's auc: 0.871488	valid_0's binary_logloss: 0.125147	valid_1's auc: 0.839671	valid_1's binary_logloss: 0.139365
    [28]	valid_0's auc: 0.87235	valid_0's binary_logloss: 0.12484	valid_1's auc: 0.840114	valid_1's binary_logloss: 0.139236
    [29]	valid_0's auc: 0.872991	valid_0's binary_logloss: 0.124593	valid_1's auc: 0.839491	valid_1's binary_logloss: 0.139271
    [30]	valid_0's auc: 0.874129	valid_0's binary_logloss: 0.124312	valid_1's auc: 0.839589	valid_1's binary_logloss: 0.13918
    [31]	valid_0's auc: 0.875305	valid_0's binary_logloss: 0.123988	valid_1's auc: 0.839441	valid_1's binary_logloss: 0.139184
    [32]	valid_0's auc: 0.875943	valid_0's binary_logloss: 0.123748	valid_1's auc: 0.839268	valid_1's binary_logloss: 0.13919
    [33]	valid_0's auc: 0.876575	valid_0's binary_logloss: 0.123484	valid_1's auc: 0.839549	valid_1's binary_logloss: 0.139075
    [34]	valid_0's auc: 0.877426	valid_0's binary_logloss: 0.123156	valid_1's auc: 0.839087	valid_1's binary_logloss: 0.139148
    [35]	valid_0's auc: 0.87822	valid_0's binary_logloss: 0.122873	valid_1's auc: 0.8389	valid_1's binary_logloss: 0.139187
    [36]	valid_0's auc: 0.878932	valid_0's binary_logloss: 0.12259	valid_1's auc: 0.838921	valid_1's binary_logloss: 0.139194
    [37]	valid_0's auc: 0.879842	valid_0's binary_logloss: 0.12233	valid_1's auc: 0.839133	valid_1's binary_logloss: 0.139161
    [38]	valid_0's auc: 0.880497	valid_0's binary_logloss: 0.12208	valid_1's auc: 0.838975	valid_1's binary_logloss: 0.139143
    [39]	valid_0's auc: 0.881056	valid_0's binary_logloss: 0.121827	valid_1's auc: 0.839037	valid_1's binary_logloss: 0.139138
    [40]	valid_0's auc: 0.881604	valid_0's binary_logloss: 0.121603	valid_1's auc: 0.839204	valid_1's binary_logloss: 0.139119
    [41]	valid_0's auc: 0.882159	valid_0's binary_logloss: 0.121355	valid_1's auc: 0.839277	valid_1's binary_logloss: 0.139091
    [42]	valid_0's auc: 0.882757	valid_0's binary_logloss: 0.121116	valid_1's auc: 0.838964	valid_1's binary_logloss: 0.139133
    [43]	valid_0's auc: 0.883143	valid_0's binary_logloss: 0.120918	valid_1's auc: 0.839024	valid_1's binary_logloss: 0.139124
    [44]	valid_0's auc: 0.883697	valid_0's binary_logloss: 0.12072	valid_1's auc: 0.838652	valid_1's binary_logloss: 0.139203
    [45]	valid_0's auc: 0.884292	valid_0's binary_logloss: 0.120482	valid_1's auc: 0.839016	valid_1's binary_logloss: 0.139124
    [46]	valid_0's auc: 0.884969	valid_0's binary_logloss: 0.120266	valid_1's auc: 0.838683	valid_1's binary_logloss: 0.139184
    [47]	valid_0's auc: 0.8853	valid_0's binary_logloss: 0.120089	valid_1's auc: 0.838624	valid_1's binary_logloss: 0.139193
    [48]	valid_0's auc: 0.885876	valid_0's binary_logloss: 0.11993	valid_1's auc: 0.838569	valid_1's binary_logloss: 0.139212
    [49]	valid_0's auc: 0.886141	valid_0's binary_logloss: 0.119757	valid_1's auc: 0.838345	valid_1's binary_logloss: 0.139288
    [50]	valid_0's auc: 0.886433	valid_0's binary_logloss: 0.119595	valid_1's auc: 0.838342	valid_1's binary_logloss: 0.139332
    [51]	valid_0's auc: 0.886975	valid_0's binary_logloss: 0.119377	valid_1's auc: 0.838335	valid_1's binary_logloss: 0.139331
    [52]	valid_0's auc: 0.887568	valid_0's binary_logloss: 0.119161	valid_1's auc: 0.838204	valid_1's binary_logloss: 0.139331
    [53]	valid_0's auc: 0.887867	valid_0's binary_logloss: 0.118974	valid_1's auc: 0.838044	valid_1's binary_logloss: 0.13936
    [54]	valid_0's auc: 0.888093	valid_0's binary_logloss: 0.118834	valid_1's auc: 0.838137	valid_1's binary_logloss: 0.13935
    [55]	valid_0's auc: 0.888289	valid_0's binary_logloss: 0.118675	valid_1's auc: 0.837878	valid_1's binary_logloss: 0.139392
    [56]	valid_0's auc: 0.888615	valid_0's binary_logloss: 0.118561	valid_1's auc: 0.837776	valid_1's binary_logloss: 0.139418
    [57]	valid_0's auc: 0.889157	valid_0's binary_logloss: 0.118369	valid_1's auc: 0.837585	valid_1's binary_logloss: 0.139447
    [58]	valid_0's auc: 0.889659	valid_0's binary_logloss: 0.11819	valid_1's auc: 0.837789	valid_1's binary_logloss: 0.139431
    Early stopping, best iteration is:
    [28]	valid_0's auc: 0.87235	valid_0's binary_logloss: 0.12484	valid_1's auc: 0.840114	valid_1's binary_logloss: 0.139236
    [1]	valid_0's auc: 0.832891	valid_0's binary_logloss: 0.155302	valid_1's auc: 0.818851	valid_1's binary_logloss: 0.164826
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.84519	valid_0's binary_logloss: 0.149727	valid_1's auc: 0.827144	valid_1's binary_logloss: 0.159879
    [3]	valid_0's auc: 0.848018	valid_0's binary_logloss: 0.145627	valid_1's auc: 0.826851	valid_1's binary_logloss: 0.15631
    [4]	valid_0's auc: 0.851096	valid_0's binary_logloss: 0.142423	valid_1's auc: 0.83073	valid_1's binary_logloss: 0.1534
    [5]	valid_0's auc: 0.854735	valid_0's binary_logloss: 0.139746	valid_1's auc: 0.832753	valid_1's binary_logloss: 0.151136
    [6]	valid_0's auc: 0.856928	valid_0's binary_logloss: 0.137509	valid_1's auc: 0.835605	valid_1's binary_logloss: 0.14924
    [7]	valid_0's auc: 0.859448	valid_0's binary_logloss: 0.135575	valid_1's auc: 0.835612	valid_1's binary_logloss: 0.147799
    [8]	valid_0's auc: 0.861685	valid_0's binary_logloss: 0.133953	valid_1's auc: 0.834408	valid_1's binary_logloss: 0.146634
    [9]	valid_0's auc: 0.863391	valid_0's binary_logloss: 0.132468	valid_1's auc: 0.835623	valid_1's binary_logloss: 0.145549
    [10]	valid_0's auc: 0.865858	valid_0's binary_logloss: 0.131185	valid_1's auc: 0.83487	valid_1's binary_logloss: 0.144745
    [11]	valid_0's auc: 0.867134	valid_0's binary_logloss: 0.130116	valid_1's auc: 0.834692	valid_1's binary_logloss: 0.14411
    [12]	valid_0's auc: 0.868217	valid_0's binary_logloss: 0.129097	valid_1's auc: 0.834746	valid_1's binary_logloss: 0.143527
    [13]	valid_0's auc: 0.87073	valid_0's binary_logloss: 0.128129	valid_1's auc: 0.833582	valid_1's binary_logloss: 0.143122
    [14]	valid_0's auc: 0.872621	valid_0's binary_logloss: 0.12721	valid_1's auc: 0.833205	valid_1's binary_logloss: 0.142745
    [15]	valid_0's auc: 0.874007	valid_0's binary_logloss: 0.126363	valid_1's auc: 0.83246	valid_1's binary_logloss: 0.142489
    [16]	valid_0's auc: 0.875141	valid_0's binary_logloss: 0.125606	valid_1's auc: 0.831958	valid_1's binary_logloss: 0.142275
    [17]	valid_0's auc: 0.876061	valid_0's binary_logloss: 0.124928	valid_1's auc: 0.831586	valid_1's binary_logloss: 0.142141
    [18]	valid_0's auc: 0.876982	valid_0's binary_logloss: 0.124313	valid_1's auc: 0.830954	valid_1's binary_logloss: 0.142066
    [19]	valid_0's auc: 0.877885	valid_0's binary_logloss: 0.123709	valid_1's auc: 0.830572	valid_1's binary_logloss: 0.14196
    [20]	valid_0's auc: 0.879378	valid_0's binary_logloss: 0.123088	valid_1's auc: 0.830076	valid_1's binary_logloss: 0.14196
    [21]	valid_0's auc: 0.880647	valid_0's binary_logloss: 0.122488	valid_1's auc: 0.830109	valid_1's binary_logloss: 0.141858
    [22]	valid_0's auc: 0.881614	valid_0's binary_logloss: 0.121973	valid_1's auc: 0.829735	valid_1's binary_logloss: 0.141822
    [23]	valid_0's auc: 0.882402	valid_0's binary_logloss: 0.121554	valid_1's auc: 0.829254	valid_1's binary_logloss: 0.141805
    [24]	valid_0's auc: 0.883011	valid_0's binary_logloss: 0.121078	valid_1's auc: 0.829054	valid_1's binary_logloss: 0.14178
    [25]	valid_0's auc: 0.884627	valid_0's binary_logloss: 0.120587	valid_1's auc: 0.82942	valid_1's binary_logloss: 0.141653
    [26]	valid_0's auc: 0.885304	valid_0's binary_logloss: 0.120169	valid_1's auc: 0.828716	valid_1's binary_logloss: 0.141755
    [27]	valid_0's auc: 0.88664	valid_0's binary_logloss: 0.119673	valid_1's auc: 0.828869	valid_1's binary_logloss: 0.141682
    [28]	valid_0's auc: 0.887143	valid_0's binary_logloss: 0.119308	valid_1's auc: 0.828987	valid_1's binary_logloss: 0.141649
    [29]	valid_0's auc: 0.88825	valid_0's binary_logloss: 0.1189	valid_1's auc: 0.829075	valid_1's binary_logloss: 0.141601
    [30]	valid_0's auc: 0.889081	valid_0's binary_logloss: 0.118531	valid_1's auc: 0.828871	valid_1's binary_logloss: 0.141605
    [31]	valid_0's auc: 0.890195	valid_0's binary_logloss: 0.118117	valid_1's auc: 0.828972	valid_1's binary_logloss: 0.141605
    [32]	valid_0's auc: 0.890928	valid_0's binary_logloss: 0.117735	valid_1's auc: 0.827969	valid_1's binary_logloss: 0.141796
    [33]	valid_0's auc: 0.891505	valid_0's binary_logloss: 0.117389	valid_1's auc: 0.827611	valid_1's binary_logloss: 0.141916
    [34]	valid_0's auc: 0.892223	valid_0's binary_logloss: 0.11707	valid_1's auc: 0.827019	valid_1's binary_logloss: 0.142051
    [35]	valid_0's auc: 0.892825	valid_0's binary_logloss: 0.116751	valid_1's auc: 0.826865	valid_1's binary_logloss: 0.142116
    [36]	valid_0's auc: 0.893984	valid_0's binary_logloss: 0.116353	valid_1's auc: 0.827203	valid_1's binary_logloss: 0.14207
    [37]	valid_0's auc: 0.89456	valid_0's binary_logloss: 0.11603	valid_1's auc: 0.827292	valid_1's binary_logloss: 0.142005
    [38]	valid_0's auc: 0.89511	valid_0's binary_logloss: 0.115713	valid_1's auc: 0.827214	valid_1's binary_logloss: 0.14206
    [39]	valid_0's auc: 0.895738	valid_0's binary_logloss: 0.115415	valid_1's auc: 0.82695	valid_1's binary_logloss: 0.142162
    Early stopping, best iteration is:
    [9]	valid_0's auc: 0.863391	valid_0's binary_logloss: 0.132468	valid_1's auc: 0.835623	valid_1's binary_logloss: 0.145549
    [1]	valid_0's auc: 0.833054	valid_0's binary_logloss: 0.15572	valid_1's auc: 0.817048	valid_1's binary_logloss: 0.165036
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.841397	valid_0's binary_logloss: 0.149862	valid_1's auc: 0.82157	valid_1's binary_logloss: 0.159575
    [3]	valid_0's auc: 0.849058	valid_0's binary_logloss: 0.145662	valid_1's auc: 0.829866	valid_1's binary_logloss: 0.155774
    [4]	valid_0's auc: 0.854301	valid_0's binary_logloss: 0.142356	valid_1's auc: 0.832415	valid_1's binary_logloss: 0.152936
    [5]	valid_0's auc: 0.858045	valid_0's binary_logloss: 0.139697	valid_1's auc: 0.834554	valid_1's binary_logloss: 0.150635
    [6]	valid_0's auc: 0.860767	valid_0's binary_logloss: 0.137458	valid_1's auc: 0.834885	valid_1's binary_logloss: 0.148761
    [7]	valid_0's auc: 0.863011	valid_0's binary_logloss: 0.135522	valid_1's auc: 0.835812	valid_1's binary_logloss: 0.147245
    [8]	valid_0's auc: 0.864923	valid_0's binary_logloss: 0.133792	valid_1's auc: 0.836656	valid_1's binary_logloss: 0.145923
    [9]	valid_0's auc: 0.865706	valid_0's binary_logloss: 0.13236	valid_1's auc: 0.836912	valid_1's binary_logloss: 0.144867
    [10]	valid_0's auc: 0.867693	valid_0's binary_logloss: 0.131066	valid_1's auc: 0.837266	valid_1's binary_logloss: 0.143895
    [11]	valid_0's auc: 0.868596	valid_0's binary_logloss: 0.129937	valid_1's auc: 0.836466	valid_1's binary_logloss: 0.143255
    [12]	valid_0's auc: 0.87012	valid_0's binary_logloss: 0.128904	valid_1's auc: 0.836589	valid_1's binary_logloss: 0.142728
    [13]	valid_0's auc: 0.871703	valid_0's binary_logloss: 0.127913	valid_1's auc: 0.836567	valid_1's binary_logloss: 0.142105
    [14]	valid_0's auc: 0.873468	valid_0's binary_logloss: 0.126983	valid_1's auc: 0.835538	valid_1's binary_logloss: 0.141771
    [15]	valid_0's auc: 0.874839	valid_0's binary_logloss: 0.126147	valid_1's auc: 0.835363	valid_1's binary_logloss: 0.141464
    [16]	valid_0's auc: 0.876399	valid_0's binary_logloss: 0.125331	valid_1's auc: 0.83478	valid_1's binary_logloss: 0.141245
    [17]	valid_0's auc: 0.877465	valid_0's binary_logloss: 0.124655	valid_1's auc: 0.834621	valid_1's binary_logloss: 0.141028
    [18]	valid_0's auc: 0.878935	valid_0's binary_logloss: 0.123944	valid_1's auc: 0.834165	valid_1's binary_logloss: 0.140935
    [19]	valid_0's auc: 0.88046	valid_0's binary_logloss: 0.123313	valid_1's auc: 0.834629	valid_1's binary_logloss: 0.140738
    [20]	valid_0's auc: 0.881517	valid_0's binary_logloss: 0.12269	valid_1's auc: 0.8347	valid_1's binary_logloss: 0.140611
    [21]	valid_0's auc: 0.882464	valid_0's binary_logloss: 0.122095	valid_1's auc: 0.834656	valid_1's binary_logloss: 0.140487
    [22]	valid_0's auc: 0.883744	valid_0's binary_logloss: 0.121504	valid_1's auc: 0.834562	valid_1's binary_logloss: 0.140328
    [23]	valid_0's auc: 0.885301	valid_0's binary_logloss: 0.12091	valid_1's auc: 0.835278	valid_1's binary_logloss: 0.140199
    [24]	valid_0's auc: 0.886266	valid_0's binary_logloss: 0.120437	valid_1's auc: 0.835728	valid_1's binary_logloss: 0.140094
    [25]	valid_0's auc: 0.88755	valid_0's binary_logloss: 0.119931	valid_1's auc: 0.836199	valid_1's binary_logloss: 0.140076
    [26]	valid_0's auc: 0.888525	valid_0's binary_logloss: 0.119473	valid_1's auc: 0.836708	valid_1's binary_logloss: 0.139945
    [27]	valid_0's auc: 0.889589	valid_0's binary_logloss: 0.119012	valid_1's auc: 0.836951	valid_1's binary_logloss: 0.139843
    [28]	valid_0's auc: 0.890552	valid_0's binary_logloss: 0.118602	valid_1's auc: 0.836524	valid_1's binary_logloss: 0.139871
    [29]	valid_0's auc: 0.891402	valid_0's binary_logloss: 0.118166	valid_1's auc: 0.836264	valid_1's binary_logloss: 0.139884
    [30]	valid_0's auc: 0.891982	valid_0's binary_logloss: 0.117805	valid_1's auc: 0.835959	valid_1's binary_logloss: 0.139937
    [31]	valid_0's auc: 0.893185	valid_0's binary_logloss: 0.117392	valid_1's auc: 0.836384	valid_1's binary_logloss: 0.13992
    [32]	valid_0's auc: 0.894065	valid_0's binary_logloss: 0.117017	valid_1's auc: 0.836341	valid_1's binary_logloss: 0.139888
    [33]	valid_0's auc: 0.894791	valid_0's binary_logloss: 0.116671	valid_1's auc: 0.836753	valid_1's binary_logloss: 0.139812
    [34]	valid_0's auc: 0.895313	valid_0's binary_logloss: 0.116321	valid_1's auc: 0.836733	valid_1's binary_logloss: 0.139826
    [35]	valid_0's auc: 0.895876	valid_0's binary_logloss: 0.116039	valid_1's auc: 0.836245	valid_1's binary_logloss: 0.139883
    [36]	valid_0's auc: 0.896909	valid_0's binary_logloss: 0.115684	valid_1's auc: 0.836079	valid_1's binary_logloss: 0.139912
    [37]	valid_0's auc: 0.897427	valid_0's binary_logloss: 0.115388	valid_1's auc: 0.835564	valid_1's binary_logloss: 0.140024
    [38]	valid_0's auc: 0.898442	valid_0's binary_logloss: 0.115006	valid_1's auc: 0.835612	valid_1's binary_logloss: 0.140075
    [39]	valid_0's auc: 0.899304	valid_0's binary_logloss: 0.114592	valid_1's auc: 0.836273	valid_1's binary_logloss: 0.139974
    [40]	valid_0's auc: 0.89974	valid_0's binary_logloss: 0.11432	valid_1's auc: 0.836096	valid_1's binary_logloss: 0.140042
    Early stopping, best iteration is:
    [10]	valid_0's auc: 0.867693	valid_0's binary_logloss: 0.131066	valid_1's auc: 0.837266	valid_1's binary_logloss: 0.143895
    [1]	valid_0's auc: 0.830643	valid_0's binary_logloss: 0.155759	valid_1's auc: 0.816734	valid_1's binary_logloss: 0.164985
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.839353	valid_0's binary_logloss: 0.149977	valid_1's auc: 0.822571	valid_1's binary_logloss: 0.159808
    [3]	valid_0's auc: 0.847366	valid_0's binary_logloss: 0.145866	valid_1's auc: 0.829312	valid_1's binary_logloss: 0.156171
    [4]	valid_0's auc: 0.850911	valid_0's binary_logloss: 0.14247	valid_1's auc: 0.830848	valid_1's binary_logloss: 0.153328
    [5]	valid_0's auc: 0.854674	valid_0's binary_logloss: 0.139764	valid_1's auc: 0.833041	valid_1's binary_logloss: 0.151023
    [6]	valid_0's auc: 0.856722	valid_0's binary_logloss: 0.1375	valid_1's auc: 0.834264	valid_1's binary_logloss: 0.149166
    [7]	valid_0's auc: 0.858253	valid_0's binary_logloss: 0.135713	valid_1's auc: 0.834998	valid_1's binary_logloss: 0.147631
    [8]	valid_0's auc: 0.859768	valid_0's binary_logloss: 0.134063	valid_1's auc: 0.835678	valid_1's binary_logloss: 0.146384
    [9]	valid_0's auc: 0.86262	valid_0's binary_logloss: 0.132622	valid_1's auc: 0.836272	valid_1's binary_logloss: 0.145313
    [10]	valid_0's auc: 0.864631	valid_0's binary_logloss: 0.131324	valid_1's auc: 0.835827	valid_1's binary_logloss: 0.144553
    [11]	valid_0's auc: 0.866805	valid_0's binary_logloss: 0.130172	valid_1's auc: 0.835375	valid_1's binary_logloss: 0.143933
    [12]	valid_0's auc: 0.868266	valid_0's binary_logloss: 0.129101	valid_1's auc: 0.835951	valid_1's binary_logloss: 0.143342
    [13]	valid_0's auc: 0.870762	valid_0's binary_logloss: 0.128144	valid_1's auc: 0.83626	valid_1's binary_logloss: 0.142813
    [14]	valid_0's auc: 0.872747	valid_0's binary_logloss: 0.127222	valid_1's auc: 0.835864	valid_1's binary_logloss: 0.142466
    [15]	valid_0's auc: 0.874158	valid_0's binary_logloss: 0.126428	valid_1's auc: 0.83548	valid_1's binary_logloss: 0.142108
    [16]	valid_0's auc: 0.875931	valid_0's binary_logloss: 0.125651	valid_1's auc: 0.836367	valid_1's binary_logloss: 0.141684
    [17]	valid_0's auc: 0.876854	valid_0's binary_logloss: 0.124918	valid_1's auc: 0.835689	valid_1's binary_logloss: 0.141524
    [18]	valid_0's auc: 0.878211	valid_0's binary_logloss: 0.124197	valid_1's auc: 0.835893	valid_1's binary_logloss: 0.141285
    [19]	valid_0's auc: 0.879125	valid_0's binary_logloss: 0.123553	valid_1's auc: 0.835877	valid_1's binary_logloss: 0.141128
    [20]	valid_0's auc: 0.880489	valid_0's binary_logloss: 0.122856	valid_1's auc: 0.835385	valid_1's binary_logloss: 0.141032
    [21]	valid_0's auc: 0.881696	valid_0's binary_logloss: 0.122219	valid_1's auc: 0.835822	valid_1's binary_logloss: 0.140843
    [22]	valid_0's auc: 0.882257	valid_0's binary_logloss: 0.121726	valid_1's auc: 0.835849	valid_1's binary_logloss: 0.140761
    [23]	valid_0's auc: 0.883635	valid_0's binary_logloss: 0.121206	valid_1's auc: 0.836223	valid_1's binary_logloss: 0.140607
    [24]	valid_0's auc: 0.884533	valid_0's binary_logloss: 0.120734	valid_1's auc: 0.836473	valid_1's binary_logloss: 0.14049
    [25]	valid_0's auc: 0.885234	valid_0's binary_logloss: 0.120268	valid_1's auc: 0.836722	valid_1's binary_logloss: 0.140403
    [26]	valid_0's auc: 0.886292	valid_0's binary_logloss: 0.119794	valid_1's auc: 0.836549	valid_1's binary_logloss: 0.140423
    [27]	valid_0's auc: 0.887064	valid_0's binary_logloss: 0.119366	valid_1's auc: 0.836155	valid_1's binary_logloss: 0.140447
    [28]	valid_0's auc: 0.887621	valid_0's binary_logloss: 0.119008	valid_1's auc: 0.835594	valid_1's binary_logloss: 0.140532
    [29]	valid_0's auc: 0.888965	valid_0's binary_logloss: 0.118547	valid_1's auc: 0.835464	valid_1's binary_logloss: 0.140508
    [30]	valid_0's auc: 0.889898	valid_0's binary_logloss: 0.118139	valid_1's auc: 0.83577	valid_1's binary_logloss: 0.140461
    [31]	valid_0's auc: 0.890896	valid_0's binary_logloss: 0.117734	valid_1's auc: 0.835475	valid_1's binary_logloss: 0.140463
    [32]	valid_0's auc: 0.892374	valid_0's binary_logloss: 0.1173	valid_1's auc: 0.835364	valid_1's binary_logloss: 0.140506
    [33]	valid_0's auc: 0.893164	valid_0's binary_logloss: 0.116978	valid_1's auc: 0.835865	valid_1's binary_logloss: 0.14041
    [34]	valid_0's auc: 0.893848	valid_0's binary_logloss: 0.11662	valid_1's auc: 0.836021	valid_1's binary_logloss: 0.140353
    [35]	valid_0's auc: 0.894232	valid_0's binary_logloss: 0.116323	valid_1's auc: 0.8359	valid_1's binary_logloss: 0.140396
    [36]	valid_0's auc: 0.895003	valid_0's binary_logloss: 0.115986	valid_1's auc: 0.835855	valid_1's binary_logloss: 0.140416
    [37]	valid_0's auc: 0.895898	valid_0's binary_logloss: 0.115609	valid_1's auc: 0.836185	valid_1's binary_logloss: 0.140369
    [38]	valid_0's auc: 0.896459	valid_0's binary_logloss: 0.11527	valid_1's auc: 0.835754	valid_1's binary_logloss: 0.140443
    [39]	valid_0's auc: 0.897377	valid_0's binary_logloss: 0.114873	valid_1's auc: 0.835638	valid_1's binary_logloss: 0.140474
    [40]	valid_0's auc: 0.89776	valid_0's binary_logloss: 0.114588	valid_1's auc: 0.835639	valid_1's binary_logloss: 0.140491
    [41]	valid_0's auc: 0.898583	valid_0's binary_logloss: 0.114302	valid_1's auc: 0.835705	valid_1's binary_logloss: 0.140506
    [42]	valid_0's auc: 0.899197	valid_0's binary_logloss: 0.113975	valid_1's auc: 0.835052	valid_1's binary_logloss: 0.14064
    [43]	valid_0's auc: 0.899803	valid_0's binary_logloss: 0.113654	valid_1's auc: 0.835035	valid_1's binary_logloss: 0.140691
    [44]	valid_0's auc: 0.900641	valid_0's binary_logloss: 0.113388	valid_1's auc: 0.835214	valid_1's binary_logloss: 0.140703
    [45]	valid_0's auc: 0.900962	valid_0's binary_logloss: 0.113098	valid_1's auc: 0.835276	valid_1's binary_logloss: 0.140695
    [46]	valid_0's auc: 0.901584	valid_0's binary_logloss: 0.112771	valid_1's auc: 0.83495	valid_1's binary_logloss: 0.140754
    [47]	valid_0's auc: 0.902256	valid_0's binary_logloss: 0.112493	valid_1's auc: 0.835639	valid_1's binary_logloss: 0.14064
    [48]	valid_0's auc: 0.902688	valid_0's binary_logloss: 0.112198	valid_1's auc: 0.835495	valid_1's binary_logloss: 0.140691
    [49]	valid_0's auc: 0.902922	valid_0's binary_logloss: 0.111944	valid_1's auc: 0.835281	valid_1's binary_logloss: 0.140819
    [50]	valid_0's auc: 0.903747	valid_0's binary_logloss: 0.111595	valid_1's auc: 0.835359	valid_1's binary_logloss: 0.140811
    [51]	valid_0's auc: 0.904427	valid_0's binary_logloss: 0.111354	valid_1's auc: 0.835245	valid_1's binary_logloss: 0.140873
    [52]	valid_0's auc: 0.90467	valid_0's binary_logloss: 0.111111	valid_1's auc: 0.835057	valid_1's binary_logloss: 0.140993
    [53]	valid_0's auc: 0.904868	valid_0's binary_logloss: 0.110853	valid_1's auc: 0.834751	valid_1's binary_logloss: 0.14108
    [54]	valid_0's auc: 0.905166	valid_0's binary_logloss: 0.110627	valid_1's auc: 0.83411	valid_1's binary_logloss: 0.141282
    [55]	valid_0's auc: 0.905665	valid_0's binary_logloss: 0.110375	valid_1's auc: 0.833739	valid_1's binary_logloss: 0.141413
    Early stopping, best iteration is:
    [25]	valid_0's auc: 0.885234	valid_0's binary_logloss: 0.120268	valid_1's auc: 0.836722	valid_1's binary_logloss: 0.140403
    [1]	valid_0's auc: 0.832891	valid_0's binary_logloss: 0.155302	valid_1's auc: 0.818851	valid_1's binary_logloss: 0.164826
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.84519	valid_0's binary_logloss: 0.149727	valid_1's auc: 0.827144	valid_1's binary_logloss: 0.159879
    [3]	valid_0's auc: 0.848018	valid_0's binary_logloss: 0.145627	valid_1's auc: 0.826851	valid_1's binary_logloss: 0.15631
    [4]	valid_0's auc: 0.851096	valid_0's binary_logloss: 0.142423	valid_1's auc: 0.83073	valid_1's binary_logloss: 0.1534
    [5]	valid_0's auc: 0.854735	valid_0's binary_logloss: 0.139746	valid_1's auc: 0.832753	valid_1's binary_logloss: 0.151136
    [6]	valid_0's auc: 0.856928	valid_0's binary_logloss: 0.137509	valid_1's auc: 0.835605	valid_1's binary_logloss: 0.14924
    [7]	valid_0's auc: 0.859448	valid_0's binary_logloss: 0.135575	valid_1's auc: 0.835612	valid_1's binary_logloss: 0.147799
    [8]	valid_0's auc: 0.861685	valid_0's binary_logloss: 0.133953	valid_1's auc: 0.834408	valid_1's binary_logloss: 0.146634
    [9]	valid_0's auc: 0.863391	valid_0's binary_logloss: 0.132468	valid_1's auc: 0.835623	valid_1's binary_logloss: 0.145549
    [10]	valid_0's auc: 0.865858	valid_0's binary_logloss: 0.131185	valid_1's auc: 0.83487	valid_1's binary_logloss: 0.144745
    [11]	valid_0's auc: 0.867134	valid_0's binary_logloss: 0.130116	valid_1's auc: 0.834692	valid_1's binary_logloss: 0.14411
    [12]	valid_0's auc: 0.868217	valid_0's binary_logloss: 0.129097	valid_1's auc: 0.834746	valid_1's binary_logloss: 0.143527
    [13]	valid_0's auc: 0.87073	valid_0's binary_logloss: 0.128129	valid_1's auc: 0.833582	valid_1's binary_logloss: 0.143122
    [14]	valid_0's auc: 0.872621	valid_0's binary_logloss: 0.12721	valid_1's auc: 0.833205	valid_1's binary_logloss: 0.142745
    [15]	valid_0's auc: 0.874007	valid_0's binary_logloss: 0.126363	valid_1's auc: 0.83246	valid_1's binary_logloss: 0.142489
    [16]	valid_0's auc: 0.875141	valid_0's binary_logloss: 0.125606	valid_1's auc: 0.831958	valid_1's binary_logloss: 0.142275
    [17]	valid_0's auc: 0.876061	valid_0's binary_logloss: 0.124928	valid_1's auc: 0.831586	valid_1's binary_logloss: 0.142141
    [18]	valid_0's auc: 0.876982	valid_0's binary_logloss: 0.124313	valid_1's auc: 0.830954	valid_1's binary_logloss: 0.142066
    [19]	valid_0's auc: 0.877885	valid_0's binary_logloss: 0.123709	valid_1's auc: 0.830572	valid_1's binary_logloss: 0.14196
    [20]	valid_0's auc: 0.879378	valid_0's binary_logloss: 0.123088	valid_1's auc: 0.830076	valid_1's binary_logloss: 0.14196
    [21]	valid_0's auc: 0.880647	valid_0's binary_logloss: 0.122488	valid_1's auc: 0.830109	valid_1's binary_logloss: 0.141858
    [22]	valid_0's auc: 0.881614	valid_0's binary_logloss: 0.121973	valid_1's auc: 0.829735	valid_1's binary_logloss: 0.141822
    [23]	valid_0's auc: 0.882402	valid_0's binary_logloss: 0.121554	valid_1's auc: 0.829254	valid_1's binary_logloss: 0.141805
    [24]	valid_0's auc: 0.883011	valid_0's binary_logloss: 0.121078	valid_1's auc: 0.829054	valid_1's binary_logloss: 0.14178
    [25]	valid_0's auc: 0.884627	valid_0's binary_logloss: 0.120587	valid_1's auc: 0.82942	valid_1's binary_logloss: 0.141653
    [26]	valid_0's auc: 0.885304	valid_0's binary_logloss: 0.120169	valid_1's auc: 0.828716	valid_1's binary_logloss: 0.141755
    [27]	valid_0's auc: 0.88664	valid_0's binary_logloss: 0.119673	valid_1's auc: 0.828869	valid_1's binary_logloss: 0.141682
    [28]	valid_0's auc: 0.887143	valid_0's binary_logloss: 0.119308	valid_1's auc: 0.828987	valid_1's binary_logloss: 0.141649
    [29]	valid_0's auc: 0.88825	valid_0's binary_logloss: 0.1189	valid_1's auc: 0.829075	valid_1's binary_logloss: 0.141601
    [30]	valid_0's auc: 0.889081	valid_0's binary_logloss: 0.118531	valid_1's auc: 0.828871	valid_1's binary_logloss: 0.141605
    [31]	valid_0's auc: 0.890195	valid_0's binary_logloss: 0.118117	valid_1's auc: 0.828972	valid_1's binary_logloss: 0.141605
    [32]	valid_0's auc: 0.890928	valid_0's binary_logloss: 0.117735	valid_1's auc: 0.827969	valid_1's binary_logloss: 0.141796
    [33]	valid_0's auc: 0.891505	valid_0's binary_logloss: 0.117389	valid_1's auc: 0.827611	valid_1's binary_logloss: 0.141916
    [34]	valid_0's auc: 0.892223	valid_0's binary_logloss: 0.11707	valid_1's auc: 0.827019	valid_1's binary_logloss: 0.142051
    [35]	valid_0's auc: 0.892825	valid_0's binary_logloss: 0.116751	valid_1's auc: 0.826865	valid_1's binary_logloss: 0.142116
    [36]	valid_0's auc: 0.893984	valid_0's binary_logloss: 0.116353	valid_1's auc: 0.827203	valid_1's binary_logloss: 0.14207
    [37]	valid_0's auc: 0.89456	valid_0's binary_logloss: 0.11603	valid_1's auc: 0.827292	valid_1's binary_logloss: 0.142005
    [38]	valid_0's auc: 0.89511	valid_0's binary_logloss: 0.115713	valid_1's auc: 0.827214	valid_1's binary_logloss: 0.14206
    [39]	valid_0's auc: 0.895738	valid_0's binary_logloss: 0.115415	valid_1's auc: 0.82695	valid_1's binary_logloss: 0.142162
    Early stopping, best iteration is:
    [9]	valid_0's auc: 0.863391	valid_0's binary_logloss: 0.132468	valid_1's auc: 0.835623	valid_1's binary_logloss: 0.145549
    [1]	valid_0's auc: 0.833054	valid_0's binary_logloss: 0.15572	valid_1's auc: 0.817048	valid_1's binary_logloss: 0.165036
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.841397	valid_0's binary_logloss: 0.149862	valid_1's auc: 0.82157	valid_1's binary_logloss: 0.159575
    [3]	valid_0's auc: 0.849058	valid_0's binary_logloss: 0.145662	valid_1's auc: 0.829866	valid_1's binary_logloss: 0.155774
    [4]	valid_0's auc: 0.854301	valid_0's binary_logloss: 0.142356	valid_1's auc: 0.832415	valid_1's binary_logloss: 0.152936
    [5]	valid_0's auc: 0.858045	valid_0's binary_logloss: 0.139697	valid_1's auc: 0.834554	valid_1's binary_logloss: 0.150635
    [6]	valid_0's auc: 0.860767	valid_0's binary_logloss: 0.137458	valid_1's auc: 0.834885	valid_1's binary_logloss: 0.148761
    [7]	valid_0's auc: 0.863011	valid_0's binary_logloss: 0.135522	valid_1's auc: 0.835812	valid_1's binary_logloss: 0.147245
    [8]	valid_0's auc: 0.864923	valid_0's binary_logloss: 0.133792	valid_1's auc: 0.836656	valid_1's binary_logloss: 0.145923
    [9]	valid_0's auc: 0.865706	valid_0's binary_logloss: 0.13236	valid_1's auc: 0.836912	valid_1's binary_logloss: 0.144867
    [10]	valid_0's auc: 0.867693	valid_0's binary_logloss: 0.131066	valid_1's auc: 0.837266	valid_1's binary_logloss: 0.143895
    [11]	valid_0's auc: 0.868596	valid_0's binary_logloss: 0.129937	valid_1's auc: 0.836466	valid_1's binary_logloss: 0.143255
    [12]	valid_0's auc: 0.87012	valid_0's binary_logloss: 0.128904	valid_1's auc: 0.836589	valid_1's binary_logloss: 0.142728
    [13]	valid_0's auc: 0.871703	valid_0's binary_logloss: 0.127913	valid_1's auc: 0.836567	valid_1's binary_logloss: 0.142105
    [14]	valid_0's auc: 0.873468	valid_0's binary_logloss: 0.126983	valid_1's auc: 0.835538	valid_1's binary_logloss: 0.141771
    [15]	valid_0's auc: 0.874839	valid_0's binary_logloss: 0.126147	valid_1's auc: 0.835363	valid_1's binary_logloss: 0.141464
    [16]	valid_0's auc: 0.876399	valid_0's binary_logloss: 0.125331	valid_1's auc: 0.83478	valid_1's binary_logloss: 0.141245
    [17]	valid_0's auc: 0.877465	valid_0's binary_logloss: 0.124655	valid_1's auc: 0.834621	valid_1's binary_logloss: 0.141028
    [18]	valid_0's auc: 0.878935	valid_0's binary_logloss: 0.123944	valid_1's auc: 0.834165	valid_1's binary_logloss: 0.140935
    [19]	valid_0's auc: 0.88046	valid_0's binary_logloss: 0.123313	valid_1's auc: 0.834629	valid_1's binary_logloss: 0.140738
    [20]	valid_0's auc: 0.881517	valid_0's binary_logloss: 0.12269	valid_1's auc: 0.8347	valid_1's binary_logloss: 0.140611
    [21]	valid_0's auc: 0.882464	valid_0's binary_logloss: 0.122095	valid_1's auc: 0.834656	valid_1's binary_logloss: 0.140487
    [22]	valid_0's auc: 0.883744	valid_0's binary_logloss: 0.121504	valid_1's auc: 0.834562	valid_1's binary_logloss: 0.140328
    [23]	valid_0's auc: 0.885301	valid_0's binary_logloss: 0.12091	valid_1's auc: 0.835278	valid_1's binary_logloss: 0.140199
    [24]	valid_0's auc: 0.886266	valid_0's binary_logloss: 0.120437	valid_1's auc: 0.835728	valid_1's binary_logloss: 0.140094
    [25]	valid_0's auc: 0.88755	valid_0's binary_logloss: 0.119931	valid_1's auc: 0.836199	valid_1's binary_logloss: 0.140076
    [26]	valid_0's auc: 0.888525	valid_0's binary_logloss: 0.119473	valid_1's auc: 0.836708	valid_1's binary_logloss: 0.139945
    [27]	valid_0's auc: 0.889589	valid_0's binary_logloss: 0.119012	valid_1's auc: 0.836951	valid_1's binary_logloss: 0.139843
    [28]	valid_0's auc: 0.890552	valid_0's binary_logloss: 0.118602	valid_1's auc: 0.836524	valid_1's binary_logloss: 0.139871
    [29]	valid_0's auc: 0.891402	valid_0's binary_logloss: 0.118166	valid_1's auc: 0.836264	valid_1's binary_logloss: 0.139884
    [30]	valid_0's auc: 0.891982	valid_0's binary_logloss: 0.117805	valid_1's auc: 0.835959	valid_1's binary_logloss: 0.139937
    [31]	valid_0's auc: 0.893185	valid_0's binary_logloss: 0.117392	valid_1's auc: 0.836384	valid_1's binary_logloss: 0.13992
    [32]	valid_0's auc: 0.894065	valid_0's binary_logloss: 0.117017	valid_1's auc: 0.836341	valid_1's binary_logloss: 0.139888
    [33]	valid_0's auc: 0.894791	valid_0's binary_logloss: 0.116671	valid_1's auc: 0.836753	valid_1's binary_logloss: 0.139812
    [34]	valid_0's auc: 0.895313	valid_0's binary_logloss: 0.116321	valid_1's auc: 0.836733	valid_1's binary_logloss: 0.139826
    [35]	valid_0's auc: 0.895876	valid_0's binary_logloss: 0.116039	valid_1's auc: 0.836245	valid_1's binary_logloss: 0.139883
    [36]	valid_0's auc: 0.896909	valid_0's binary_logloss: 0.115684	valid_1's auc: 0.836079	valid_1's binary_logloss: 0.139912
    [37]	valid_0's auc: 0.897427	valid_0's binary_logloss: 0.115388	valid_1's auc: 0.835564	valid_1's binary_logloss: 0.140024
    [38]	valid_0's auc: 0.898442	valid_0's binary_logloss: 0.115006	valid_1's auc: 0.835612	valid_1's binary_logloss: 0.140075
    [39]	valid_0's auc: 0.899304	valid_0's binary_logloss: 0.114592	valid_1's auc: 0.836273	valid_1's binary_logloss: 0.139974
    [40]	valid_0's auc: 0.89974	valid_0's binary_logloss: 0.11432	valid_1's auc: 0.836096	valid_1's binary_logloss: 0.140042
    Early stopping, best iteration is:
    [10]	valid_0's auc: 0.867693	valid_0's binary_logloss: 0.131066	valid_1's auc: 0.837266	valid_1's binary_logloss: 0.143895
    [1]	valid_0's auc: 0.830643	valid_0's binary_logloss: 0.155759	valid_1's auc: 0.816734	valid_1's binary_logloss: 0.164985
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.839353	valid_0's binary_logloss: 0.149977	valid_1's auc: 0.822571	valid_1's binary_logloss: 0.159808
    [3]	valid_0's auc: 0.847366	valid_0's binary_logloss: 0.145866	valid_1's auc: 0.829312	valid_1's binary_logloss: 0.156171
    [4]	valid_0's auc: 0.850911	valid_0's binary_logloss: 0.14247	valid_1's auc: 0.830848	valid_1's binary_logloss: 0.153328
    [5]	valid_0's auc: 0.854674	valid_0's binary_logloss: 0.139764	valid_1's auc: 0.833041	valid_1's binary_logloss: 0.151023
    [6]	valid_0's auc: 0.856722	valid_0's binary_logloss: 0.1375	valid_1's auc: 0.834264	valid_1's binary_logloss: 0.149166
    [7]	valid_0's auc: 0.858253	valid_0's binary_logloss: 0.135713	valid_1's auc: 0.834998	valid_1's binary_logloss: 0.147631
    [8]	valid_0's auc: 0.859768	valid_0's binary_logloss: 0.134063	valid_1's auc: 0.835678	valid_1's binary_logloss: 0.146384
    [9]	valid_0's auc: 0.86262	valid_0's binary_logloss: 0.132622	valid_1's auc: 0.836272	valid_1's binary_logloss: 0.145313
    [10]	valid_0's auc: 0.864631	valid_0's binary_logloss: 0.131324	valid_1's auc: 0.835827	valid_1's binary_logloss: 0.144553
    [11]	valid_0's auc: 0.866805	valid_0's binary_logloss: 0.130172	valid_1's auc: 0.835375	valid_1's binary_logloss: 0.143933
    [12]	valid_0's auc: 0.868266	valid_0's binary_logloss: 0.129101	valid_1's auc: 0.835951	valid_1's binary_logloss: 0.143342
    [13]	valid_0's auc: 0.870762	valid_0's binary_logloss: 0.128144	valid_1's auc: 0.83626	valid_1's binary_logloss: 0.142813
    [14]	valid_0's auc: 0.872747	valid_0's binary_logloss: 0.127222	valid_1's auc: 0.835864	valid_1's binary_logloss: 0.142466
    [15]	valid_0's auc: 0.874158	valid_0's binary_logloss: 0.126428	valid_1's auc: 0.83548	valid_1's binary_logloss: 0.142108
    [16]	valid_0's auc: 0.875931	valid_0's binary_logloss: 0.125651	valid_1's auc: 0.836367	valid_1's binary_logloss: 0.141684
    [17]	valid_0's auc: 0.876854	valid_0's binary_logloss: 0.124918	valid_1's auc: 0.835689	valid_1's binary_logloss: 0.141524
    [18]	valid_0's auc: 0.878211	valid_0's binary_logloss: 0.124197	valid_1's auc: 0.835893	valid_1's binary_logloss: 0.141285
    [19]	valid_0's auc: 0.879125	valid_0's binary_logloss: 0.123553	valid_1's auc: 0.835877	valid_1's binary_logloss: 0.141128
    [20]	valid_0's auc: 0.880489	valid_0's binary_logloss: 0.122856	valid_1's auc: 0.835385	valid_1's binary_logloss: 0.141032
    [21]	valid_0's auc: 0.881696	valid_0's binary_logloss: 0.122219	valid_1's auc: 0.835822	valid_1's binary_logloss: 0.140843
    [22]	valid_0's auc: 0.882257	valid_0's binary_logloss: 0.121726	valid_1's auc: 0.835849	valid_1's binary_logloss: 0.140761
    [23]	valid_0's auc: 0.883635	valid_0's binary_logloss: 0.121206	valid_1's auc: 0.836223	valid_1's binary_logloss: 0.140607
    [24]	valid_0's auc: 0.884533	valid_0's binary_logloss: 0.120734	valid_1's auc: 0.836473	valid_1's binary_logloss: 0.14049
    [25]	valid_0's auc: 0.885234	valid_0's binary_logloss: 0.120268	valid_1's auc: 0.836722	valid_1's binary_logloss: 0.140403
    [26]	valid_0's auc: 0.886292	valid_0's binary_logloss: 0.119794	valid_1's auc: 0.836549	valid_1's binary_logloss: 0.140423
    [27]	valid_0's auc: 0.887064	valid_0's binary_logloss: 0.119366	valid_1's auc: 0.836155	valid_1's binary_logloss: 0.140447
    [28]	valid_0's auc: 0.887621	valid_0's binary_logloss: 0.119008	valid_1's auc: 0.835594	valid_1's binary_logloss: 0.140532
    [29]	valid_0's auc: 0.888965	valid_0's binary_logloss: 0.118547	valid_1's auc: 0.835464	valid_1's binary_logloss: 0.140508
    [30]	valid_0's auc: 0.889898	valid_0's binary_logloss: 0.118139	valid_1's auc: 0.83577	valid_1's binary_logloss: 0.140461
    [31]	valid_0's auc: 0.890896	valid_0's binary_logloss: 0.117734	valid_1's auc: 0.835475	valid_1's binary_logloss: 0.140463
    [32]	valid_0's auc: 0.892374	valid_0's binary_logloss: 0.1173	valid_1's auc: 0.835364	valid_1's binary_logloss: 0.140506
    [33]	valid_0's auc: 0.893164	valid_0's binary_logloss: 0.116978	valid_1's auc: 0.835865	valid_1's binary_logloss: 0.14041
    [34]	valid_0's auc: 0.893848	valid_0's binary_logloss: 0.11662	valid_1's auc: 0.836021	valid_1's binary_logloss: 0.140353
    [35]	valid_0's auc: 0.894232	valid_0's binary_logloss: 0.116323	valid_1's auc: 0.8359	valid_1's binary_logloss: 0.140396
    [36]	valid_0's auc: 0.895003	valid_0's binary_logloss: 0.115986	valid_1's auc: 0.835855	valid_1's binary_logloss: 0.140416
    [37]	valid_0's auc: 0.895898	valid_0's binary_logloss: 0.115609	valid_1's auc: 0.836185	valid_1's binary_logloss: 0.140369
    [38]	valid_0's auc: 0.896459	valid_0's binary_logloss: 0.11527	valid_1's auc: 0.835754	valid_1's binary_logloss: 0.140443
    [39]	valid_0's auc: 0.897377	valid_0's binary_logloss: 0.114873	valid_1's auc: 0.835638	valid_1's binary_logloss: 0.140474
    [40]	valid_0's auc: 0.89776	valid_0's binary_logloss: 0.114588	valid_1's auc: 0.835639	valid_1's binary_logloss: 0.140491
    [41]	valid_0's auc: 0.898583	valid_0's binary_logloss: 0.114302	valid_1's auc: 0.835705	valid_1's binary_logloss: 0.140506
    [42]	valid_0's auc: 0.899197	valid_0's binary_logloss: 0.113975	valid_1's auc: 0.835052	valid_1's binary_logloss: 0.14064
    [43]	valid_0's auc: 0.899803	valid_0's binary_logloss: 0.113654	valid_1's auc: 0.835035	valid_1's binary_logloss: 0.140691
    [44]	valid_0's auc: 0.900641	valid_0's binary_logloss: 0.113388	valid_1's auc: 0.835214	valid_1's binary_logloss: 0.140703
    [45]	valid_0's auc: 0.900962	valid_0's binary_logloss: 0.113098	valid_1's auc: 0.835276	valid_1's binary_logloss: 0.140695
    [46]	valid_0's auc: 0.901584	valid_0's binary_logloss: 0.112771	valid_1's auc: 0.83495	valid_1's binary_logloss: 0.140754
    [47]	valid_0's auc: 0.902256	valid_0's binary_logloss: 0.112493	valid_1's auc: 0.835639	valid_1's binary_logloss: 0.14064
    [48]	valid_0's auc: 0.902688	valid_0's binary_logloss: 0.112198	valid_1's auc: 0.835495	valid_1's binary_logloss: 0.140691
    [49]	valid_0's auc: 0.902922	valid_0's binary_logloss: 0.111944	valid_1's auc: 0.835281	valid_1's binary_logloss: 0.140819
    [50]	valid_0's auc: 0.903747	valid_0's binary_logloss: 0.111595	valid_1's auc: 0.835359	valid_1's binary_logloss: 0.140811
    [51]	valid_0's auc: 0.904427	valid_0's binary_logloss: 0.111354	valid_1's auc: 0.835245	valid_1's binary_logloss: 0.140873
    [52]	valid_0's auc: 0.90467	valid_0's binary_logloss: 0.111111	valid_1's auc: 0.835057	valid_1's binary_logloss: 0.140993
    [53]	valid_0's auc: 0.904868	valid_0's binary_logloss: 0.110853	valid_1's auc: 0.834751	valid_1's binary_logloss: 0.14108
    [54]	valid_0's auc: 0.905166	valid_0's binary_logloss: 0.110627	valid_1's auc: 0.83411	valid_1's binary_logloss: 0.141282
    [55]	valid_0's auc: 0.905665	valid_0's binary_logloss: 0.110375	valid_1's auc: 0.833739	valid_1's binary_logloss: 0.141413
    Early stopping, best iteration is:
    [25]	valid_0's auc: 0.885234	valid_0's binary_logloss: 0.120268	valid_1's auc: 0.836722	valid_1's binary_logloss: 0.140403
    [1]	valid_0's auc: 0.824873	valid_0's binary_logloss: 0.156222	valid_1's auc: 0.817791	valid_1's binary_logloss: 0.165072
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.828725	valid_0's binary_logloss: 0.151244	valid_1's auc: 0.822586	valid_1's binary_logloss: 0.160253
    [3]	valid_0's auc: 0.83594	valid_0's binary_logloss: 0.147423	valid_1's auc: 0.828474	valid_1's binary_logloss: 0.156542
    [4]	valid_0's auc: 0.839489	valid_0's binary_logloss: 0.144426	valid_1's auc: 0.831396	valid_1's binary_logloss: 0.153706
    [5]	valid_0's auc: 0.843358	valid_0's binary_logloss: 0.142067	valid_1's auc: 0.833466	valid_1's binary_logloss: 0.151399
    [6]	valid_0's auc: 0.845601	valid_0's binary_logloss: 0.14009	valid_1's auc: 0.833857	valid_1's binary_logloss: 0.149488
    [7]	valid_0's auc: 0.846477	valid_0's binary_logloss: 0.138491	valid_1's auc: 0.833143	valid_1's binary_logloss: 0.148023
    [8]	valid_0's auc: 0.847725	valid_0's binary_logloss: 0.137129	valid_1's auc: 0.833971	valid_1's binary_logloss: 0.146757
    [9]	valid_0's auc: 0.848442	valid_0's binary_logloss: 0.135908	valid_1's auc: 0.835976	valid_1's binary_logloss: 0.145685
    [10]	valid_0's auc: 0.849759	valid_0's binary_logloss: 0.134781	valid_1's auc: 0.836214	valid_1's binary_logloss: 0.144769
    [11]	valid_0's auc: 0.852238	valid_0's binary_logloss: 0.133835	valid_1's auc: 0.837243	valid_1's binary_logloss: 0.143925
    [12]	valid_0's auc: 0.853743	valid_0's binary_logloss: 0.132972	valid_1's auc: 0.836647	valid_1's binary_logloss: 0.143391
    [13]	valid_0's auc: 0.854568	valid_0's binary_logloss: 0.132256	valid_1's auc: 0.837182	valid_1's binary_logloss: 0.142849
    [14]	valid_0's auc: 0.855928	valid_0's binary_logloss: 0.131554	valid_1's auc: 0.835941	valid_1's binary_logloss: 0.142474
    [15]	valid_0's auc: 0.85712	valid_0's binary_logloss: 0.130984	valid_1's auc: 0.834938	valid_1's binary_logloss: 0.142198
    [16]	valid_0's auc: 0.858721	valid_0's binary_logloss: 0.130371	valid_1's auc: 0.83561	valid_1's binary_logloss: 0.141802
    [17]	valid_0's auc: 0.859281	valid_0's binary_logloss: 0.129877	valid_1's auc: 0.835146	valid_1's binary_logloss: 0.141605
    [18]	valid_0's auc: 0.859881	valid_0's binary_logloss: 0.129417	valid_1's auc: 0.835386	valid_1's binary_logloss: 0.14132
    [19]	valid_0's auc: 0.861409	valid_0's binary_logloss: 0.128929	valid_1's auc: 0.834974	valid_1's binary_logloss: 0.141151
    [20]	valid_0's auc: 0.862574	valid_0's binary_logloss: 0.128458	valid_1's auc: 0.834949	valid_1's binary_logloss: 0.140968
    [21]	valid_0's auc: 0.863262	valid_0's binary_logloss: 0.128069	valid_1's auc: 0.834616	valid_1's binary_logloss: 0.14086
    [22]	valid_0's auc: 0.864655	valid_0's binary_logloss: 0.127684	valid_1's auc: 0.834363	valid_1's binary_logloss: 0.140766
    [23]	valid_0's auc: 0.865247	valid_0's binary_logloss: 0.127349	valid_1's auc: 0.834317	valid_1's binary_logloss: 0.140688
    [24]	valid_0's auc: 0.865882	valid_0's binary_logloss: 0.12704	valid_1's auc: 0.833543	valid_1's binary_logloss: 0.14068
    [25]	valid_0's auc: 0.867496	valid_0's binary_logloss: 0.126629	valid_1's auc: 0.834195	valid_1's binary_logloss: 0.140539
    [26]	valid_0's auc: 0.867923	valid_0's binary_logloss: 0.126353	valid_1's auc: 0.834028	valid_1's binary_logloss: 0.140506
    [27]	valid_0's auc: 0.868685	valid_0's binary_logloss: 0.126058	valid_1's auc: 0.834718	valid_1's binary_logloss: 0.140359
    [28]	valid_0's auc: 0.869304	valid_0's binary_logloss: 0.125764	valid_1's auc: 0.834935	valid_1's binary_logloss: 0.140287
    [29]	valid_0's auc: 0.870037	valid_0's binary_logloss: 0.125514	valid_1's auc: 0.834481	valid_1's binary_logloss: 0.140258
    [30]	valid_0's auc: 0.870785	valid_0's binary_logloss: 0.125254	valid_1's auc: 0.834179	valid_1's binary_logloss: 0.140275
    [31]	valid_0's auc: 0.871706	valid_0's binary_logloss: 0.124992	valid_1's auc: 0.834475	valid_1's binary_logloss: 0.140205
    [32]	valid_0's auc: 0.872582	valid_0's binary_logloss: 0.124728	valid_1's auc: 0.834353	valid_1's binary_logloss: 0.140189
    [33]	valid_0's auc: 0.873445	valid_0's binary_logloss: 0.124481	valid_1's auc: 0.834592	valid_1's binary_logloss: 0.140082
    [34]	valid_0's auc: 0.874095	valid_0's binary_logloss: 0.12426	valid_1's auc: 0.83436	valid_1's binary_logloss: 0.140101
    [35]	valid_0's auc: 0.874869	valid_0's binary_logloss: 0.123982	valid_1's auc: 0.834045	valid_1's binary_logloss: 0.140151
    [36]	valid_0's auc: 0.875446	valid_0's binary_logloss: 0.123753	valid_1's auc: 0.834073	valid_1's binary_logloss: 0.140125
    [37]	valid_0's auc: 0.875763	valid_0's binary_logloss: 0.123587	valid_1's auc: 0.833611	valid_1's binary_logloss: 0.140201
    [38]	valid_0's auc: 0.876603	valid_0's binary_logloss: 0.123335	valid_1's auc: 0.833805	valid_1's binary_logloss: 0.140159
    [39]	valid_0's auc: 0.877126	valid_0's binary_logloss: 0.123134	valid_1's auc: 0.834422	valid_1's binary_logloss: 0.140048
    [40]	valid_0's auc: 0.877575	valid_0's binary_logloss: 0.123013	valid_1's auc: 0.834343	valid_1's binary_logloss: 0.140069
    [41]	valid_0's auc: 0.87809	valid_0's binary_logloss: 0.122813	valid_1's auc: 0.834199	valid_1's binary_logloss: 0.140085
    Early stopping, best iteration is:
    [11]	valid_0's auc: 0.852238	valid_0's binary_logloss: 0.133835	valid_1's auc: 0.837243	valid_1's binary_logloss: 0.143925
    [1]	valid_0's auc: 0.821831	valid_0's binary_logloss: 0.156466	valid_1's auc: 0.817525	valid_1's binary_logloss: 0.165186
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.831974	valid_0's binary_logloss: 0.151137	valid_1's auc: 0.82532	valid_1's binary_logloss: 0.159691
    [3]	valid_0's auc: 0.839496	valid_0's binary_logloss: 0.14733	valid_1's auc: 0.831946	valid_1's binary_logloss: 0.156
    [4]	valid_0's auc: 0.843984	valid_0's binary_logloss: 0.144371	valid_1's auc: 0.834064	valid_1's binary_logloss: 0.153082
    [5]	valid_0's auc: 0.845854	valid_0's binary_logloss: 0.142024	valid_1's auc: 0.836918	valid_1's binary_logloss: 0.150735
    [6]	valid_0's auc: 0.848041	valid_0's binary_logloss: 0.140009	valid_1's auc: 0.838831	valid_1's binary_logloss: 0.148771
    [7]	valid_0's auc: 0.849655	valid_0's binary_logloss: 0.138307	valid_1's auc: 0.839111	valid_1's binary_logloss: 0.147373
    [8]	valid_0's auc: 0.85185	valid_0's binary_logloss: 0.136891	valid_1's auc: 0.838955	valid_1's binary_logloss: 0.146094
    [9]	valid_0's auc: 0.853067	valid_0's binary_logloss: 0.135655	valid_1's auc: 0.838081	valid_1's binary_logloss: 0.14516
    [10]	valid_0's auc: 0.853922	valid_0's binary_logloss: 0.134622	valid_1's auc: 0.837333	valid_1's binary_logloss: 0.144318
    [11]	valid_0's auc: 0.854729	valid_0's binary_logloss: 0.133702	valid_1's auc: 0.83725	valid_1's binary_logloss: 0.143512
    [12]	valid_0's auc: 0.856303	valid_0's binary_logloss: 0.132789	valid_1's auc: 0.837602	valid_1's binary_logloss: 0.142833
    [13]	valid_0's auc: 0.857206	valid_0's binary_logloss: 0.132038	valid_1's auc: 0.837364	valid_1's binary_logloss: 0.142245
    [14]	valid_0's auc: 0.858161	valid_0's binary_logloss: 0.131391	valid_1's auc: 0.83777	valid_1's binary_logloss: 0.141759
    [15]	valid_0's auc: 0.858975	valid_0's binary_logloss: 0.130772	valid_1's auc: 0.837831	valid_1's binary_logloss: 0.14139
    [16]	valid_0's auc: 0.859623	valid_0's binary_logloss: 0.130219	valid_1's auc: 0.837953	valid_1's binary_logloss: 0.141016
    [17]	valid_0's auc: 0.860576	valid_0's binary_logloss: 0.129684	valid_1's auc: 0.837985	valid_1's binary_logloss: 0.140713
    [18]	valid_0's auc: 0.861311	valid_0's binary_logloss: 0.129202	valid_1's auc: 0.83796	valid_1's binary_logloss: 0.140452
    [19]	valid_0's auc: 0.862347	valid_0's binary_logloss: 0.128715	valid_1's auc: 0.838506	valid_1's binary_logloss: 0.140189
    [20]	valid_0's auc: 0.86305	valid_0's binary_logloss: 0.128312	valid_1's auc: 0.837702	valid_1's binary_logloss: 0.140094
    [21]	valid_0's auc: 0.863758	valid_0's binary_logloss: 0.127907	valid_1's auc: 0.838127	valid_1's binary_logloss: 0.139858
    [22]	valid_0's auc: 0.864635	valid_0's binary_logloss: 0.127525	valid_1's auc: 0.838331	valid_1's binary_logloss: 0.139696
    [23]	valid_0's auc: 0.865866	valid_0's binary_logloss: 0.127143	valid_1's auc: 0.837841	valid_1's binary_logloss: 0.139625
    [24]	valid_0's auc: 0.867054	valid_0's binary_logloss: 0.126749	valid_1's auc: 0.838187	valid_1's binary_logloss: 0.139526
    [25]	valid_0's auc: 0.867553	valid_0's binary_logloss: 0.126476	valid_1's auc: 0.838308	valid_1's binary_logloss: 0.13949
    [26]	valid_0's auc: 0.868108	valid_0's binary_logloss: 0.126164	valid_1's auc: 0.838035	valid_1's binary_logloss: 0.139426
    [27]	valid_0's auc: 0.869014	valid_0's binary_logloss: 0.125868	valid_1's auc: 0.837545	valid_1's binary_logloss: 0.139445
    [28]	valid_0's auc: 0.869797	valid_0's binary_logloss: 0.12559	valid_1's auc: 0.837894	valid_1's binary_logloss: 0.139419
    [29]	valid_0's auc: 0.870435	valid_0's binary_logloss: 0.1253	valid_1's auc: 0.838103	valid_1's binary_logloss: 0.139321
    [30]	valid_0's auc: 0.87141	valid_0's binary_logloss: 0.125025	valid_1's auc: 0.838164	valid_1's binary_logloss: 0.139275
    [31]	valid_0's auc: 0.872143	valid_0's binary_logloss: 0.124769	valid_1's auc: 0.837843	valid_1's binary_logloss: 0.139285
    [32]	valid_0's auc: 0.872606	valid_0's binary_logloss: 0.124561	valid_1's auc: 0.837662	valid_1's binary_logloss: 0.139274
    [33]	valid_0's auc: 0.873337	valid_0's binary_logloss: 0.124346	valid_1's auc: 0.837661	valid_1's binary_logloss: 0.139284
    [34]	valid_0's auc: 0.873965	valid_0's binary_logloss: 0.124108	valid_1's auc: 0.837639	valid_1's binary_logloss: 0.139263
    [35]	valid_0's auc: 0.87457	valid_0's binary_logloss: 0.123857	valid_1's auc: 0.838159	valid_1's binary_logloss: 0.139137
    [36]	valid_0's auc: 0.874973	valid_0's binary_logloss: 0.123651	valid_1's auc: 0.838114	valid_1's binary_logloss: 0.139148
    [37]	valid_0's auc: 0.875657	valid_0's binary_logloss: 0.123447	valid_1's auc: 0.838519	valid_1's binary_logloss: 0.139109
    Early stopping, best iteration is:
    [7]	valid_0's auc: 0.849655	valid_0's binary_logloss: 0.138307	valid_1's auc: 0.839111	valid_1's binary_logloss: 0.147373
    [1]	valid_0's auc: 0.821427	valid_0's binary_logloss: 0.156592	valid_1's auc: 0.81711	valid_1's binary_logloss: 0.165273
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.827893	valid_0's binary_logloss: 0.151336	valid_1's auc: 0.820533	valid_1's binary_logloss: 0.160243
    [3]	valid_0's auc: 0.83753	valid_0's binary_logloss: 0.147487	valid_1's auc: 0.82841	valid_1's binary_logloss: 0.156547
    [4]	valid_0's auc: 0.84038	valid_0's binary_logloss: 0.144428	valid_1's auc: 0.8313	valid_1's binary_logloss: 0.153575
    [5]	valid_0's auc: 0.842945	valid_0's binary_logloss: 0.142089	valid_1's auc: 0.833579	valid_1's binary_logloss: 0.151354
    [6]	valid_0's auc: 0.843246	valid_0's binary_logloss: 0.140186	valid_1's auc: 0.833781	valid_1's binary_logloss: 0.14953
    [7]	valid_0's auc: 0.844301	valid_0's binary_logloss: 0.138471	valid_1's auc: 0.834317	valid_1's binary_logloss: 0.147954
    [8]	valid_0's auc: 0.846945	valid_0's binary_logloss: 0.137078	valid_1's auc: 0.834895	valid_1's binary_logloss: 0.146786
    [9]	valid_0's auc: 0.849381	valid_0's binary_logloss: 0.135906	valid_1's auc: 0.834922	valid_1's binary_logloss: 0.145762
    [10]	valid_0's auc: 0.850944	valid_0's binary_logloss: 0.134855	valid_1's auc: 0.835441	valid_1's binary_logloss: 0.144958
    [11]	valid_0's auc: 0.852557	valid_0's binary_logloss: 0.133895	valid_1's auc: 0.835103	valid_1's binary_logloss: 0.144293
    [12]	valid_0's auc: 0.854609	valid_0's binary_logloss: 0.133013	valid_1's auc: 0.835686	valid_1's binary_logloss: 0.143793
    [13]	valid_0's auc: 0.855817	valid_0's binary_logloss: 0.132247	valid_1's auc: 0.835296	valid_1's binary_logloss: 0.143302
    [14]	valid_0's auc: 0.857501	valid_0's binary_logloss: 0.131545	valid_1's auc: 0.836432	valid_1's binary_logloss: 0.142761
    [15]	valid_0's auc: 0.858907	valid_0's binary_logloss: 0.130878	valid_1's auc: 0.836329	valid_1's binary_logloss: 0.142383
    [16]	valid_0's auc: 0.859887	valid_0's binary_logloss: 0.130287	valid_1's auc: 0.836611	valid_1's binary_logloss: 0.141883
    [17]	valid_0's auc: 0.860889	valid_0's binary_logloss: 0.129757	valid_1's auc: 0.836848	valid_1's binary_logloss: 0.141535
    [18]	valid_0's auc: 0.861827	valid_0's binary_logloss: 0.129301	valid_1's auc: 0.837106	valid_1's binary_logloss: 0.141257
    [19]	valid_0's auc: 0.862972	valid_0's binary_logloss: 0.128826	valid_1's auc: 0.837185	valid_1's binary_logloss: 0.141043
    [20]	valid_0's auc: 0.864083	valid_0's binary_logloss: 0.128369	valid_1's auc: 0.837509	valid_1's binary_logloss: 0.140794
    [21]	valid_0's auc: 0.864747	valid_0's binary_logloss: 0.127959	valid_1's auc: 0.837888	valid_1's binary_logloss: 0.140626
    [22]	valid_0's auc: 0.865769	valid_0's binary_logloss: 0.127562	valid_1's auc: 0.837811	valid_1's binary_logloss: 0.140487
    [23]	valid_0's auc: 0.866657	valid_0's binary_logloss: 0.127217	valid_1's auc: 0.837884	valid_1's binary_logloss: 0.140328
    [24]	valid_0's auc: 0.867293	valid_0's binary_logloss: 0.126875	valid_1's auc: 0.838481	valid_1's binary_logloss: 0.140215
    [25]	valid_0's auc: 0.867983	valid_0's binary_logloss: 0.126562	valid_1's auc: 0.838239	valid_1's binary_logloss: 0.140124
    [26]	valid_0's auc: 0.868559	valid_0's binary_logloss: 0.126248	valid_1's auc: 0.837903	valid_1's binary_logloss: 0.140092
    [27]	valid_0's auc: 0.869394	valid_0's binary_logloss: 0.125936	valid_1's auc: 0.837493	valid_1's binary_logloss: 0.14006
    [28]	valid_0's auc: 0.87048	valid_0's binary_logloss: 0.125677	valid_1's auc: 0.837623	valid_1's binary_logloss: 0.140007
    [29]	valid_0's auc: 0.87105	valid_0's binary_logloss: 0.125405	valid_1's auc: 0.838216	valid_1's binary_logloss: 0.13986
    [30]	valid_0's auc: 0.871749	valid_0's binary_logloss: 0.125147	valid_1's auc: 0.838898	valid_1's binary_logloss: 0.139742
    [31]	valid_0's auc: 0.87247	valid_0's binary_logloss: 0.124907	valid_1's auc: 0.838959	valid_1's binary_logloss: 0.139727
    [32]	valid_0's auc: 0.87282	valid_0's binary_logloss: 0.124724	valid_1's auc: 0.838675	valid_1's binary_logloss: 0.139761
    [33]	valid_0's auc: 0.874106	valid_0's binary_logloss: 0.124412	valid_1's auc: 0.838893	valid_1's binary_logloss: 0.139687
    [34]	valid_0's auc: 0.874887	valid_0's binary_logloss: 0.124169	valid_1's auc: 0.838801	valid_1's binary_logloss: 0.139672
    [35]	valid_0's auc: 0.875447	valid_0's binary_logloss: 0.123934	valid_1's auc: 0.838835	valid_1's binary_logloss: 0.139667
    [36]	valid_0's auc: 0.87617	valid_0's binary_logloss: 0.123693	valid_1's auc: 0.838505	valid_1's binary_logloss: 0.139699
    [37]	valid_0's auc: 0.876793	valid_0's binary_logloss: 0.12346	valid_1's auc: 0.838104	valid_1's binary_logloss: 0.139783
    [38]	valid_0's auc: 0.877265	valid_0's binary_logloss: 0.123251	valid_1's auc: 0.838267	valid_1's binary_logloss: 0.139787
    [39]	valid_0's auc: 0.877869	valid_0's binary_logloss: 0.123018	valid_1's auc: 0.838004	valid_1's binary_logloss: 0.139806
    [40]	valid_0's auc: 0.878509	valid_0's binary_logloss: 0.122803	valid_1's auc: 0.838086	valid_1's binary_logloss: 0.139745
    [41]	valid_0's auc: 0.879077	valid_0's binary_logloss: 0.122585	valid_1's auc: 0.838538	valid_1's binary_logloss: 0.139694
    [42]	valid_0's auc: 0.879515	valid_0's binary_logloss: 0.122368	valid_1's auc: 0.838647	valid_1's binary_logloss: 0.139655
    [43]	valid_0's auc: 0.879985	valid_0's binary_logloss: 0.122166	valid_1's auc: 0.838495	valid_1's binary_logloss: 0.139653
    [44]	valid_0's auc: 0.88041	valid_0's binary_logloss: 0.121985	valid_1's auc: 0.838221	valid_1's binary_logloss: 0.139755
    [45]	valid_0's auc: 0.880907	valid_0's binary_logloss: 0.121777	valid_1's auc: 0.837981	valid_1's binary_logloss: 0.139769
    [46]	valid_0's auc: 0.881216	valid_0's binary_logloss: 0.121594	valid_1's auc: 0.838471	valid_1's binary_logloss: 0.139693
    [47]	valid_0's auc: 0.881591	valid_0's binary_logloss: 0.121422	valid_1's auc: 0.83861	valid_1's binary_logloss: 0.139687
    [48]	valid_0's auc: 0.881867	valid_0's binary_logloss: 0.121266	valid_1's auc: 0.838593	valid_1's binary_logloss: 0.139682
    [49]	valid_0's auc: 0.882285	valid_0's binary_logloss: 0.121041	valid_1's auc: 0.838317	valid_1's binary_logloss: 0.139741
    [50]	valid_0's auc: 0.882828	valid_0's binary_logloss: 0.120853	valid_1's auc: 0.838244	valid_1's binary_logloss: 0.139759
    [51]	valid_0's auc: 0.883154	valid_0's binary_logloss: 0.120688	valid_1's auc: 0.838222	valid_1's binary_logloss: 0.139803
    [52]	valid_0's auc: 0.883348	valid_0's binary_logloss: 0.120567	valid_1's auc: 0.838064	valid_1's binary_logloss: 0.139824
    [53]	valid_0's auc: 0.883583	valid_0's binary_logloss: 0.120424	valid_1's auc: 0.83788	valid_1's binary_logloss: 0.139844
    [54]	valid_0's auc: 0.884106	valid_0's binary_logloss: 0.120208	valid_1's auc: 0.837625	valid_1's binary_logloss: 0.139886
    [55]	valid_0's auc: 0.884777	valid_0's binary_logloss: 0.120039	valid_1's auc: 0.837585	valid_1's binary_logloss: 0.139902
    [56]	valid_0's auc: 0.88511	valid_0's binary_logloss: 0.11989	valid_1's auc: 0.837646	valid_1's binary_logloss: 0.139926
    [57]	valid_0's auc: 0.885365	valid_0's binary_logloss: 0.11975	valid_1's auc: 0.837639	valid_1's binary_logloss: 0.139934
    [58]	valid_0's auc: 0.885606	valid_0's binary_logloss: 0.119595	valid_1's auc: 0.837726	valid_1's binary_logloss: 0.139938
    [59]	valid_0's auc: 0.885965	valid_0's binary_logloss: 0.119403	valid_1's auc: 0.837558	valid_1's binary_logloss: 0.140007
    [60]	valid_0's auc: 0.886208	valid_0's binary_logloss: 0.119263	valid_1's auc: 0.83744	valid_1's binary_logloss: 0.140079
    [61]	valid_0's auc: 0.886458	valid_0's binary_logloss: 0.119118	valid_1's auc: 0.837349	valid_1's binary_logloss: 0.140059
    Early stopping, best iteration is:
    [31]	valid_0's auc: 0.87247	valid_0's binary_logloss: 0.124907	valid_1's auc: 0.838959	valid_1's binary_logloss: 0.139727
    [1]	valid_0's auc: 0.824873	valid_0's binary_logloss: 0.156222	valid_1's auc: 0.817791	valid_1's binary_logloss: 0.165072
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.828725	valid_0's binary_logloss: 0.151244	valid_1's auc: 0.822586	valid_1's binary_logloss: 0.160253
    [3]	valid_0's auc: 0.83594	valid_0's binary_logloss: 0.147423	valid_1's auc: 0.828474	valid_1's binary_logloss: 0.156542
    [4]	valid_0's auc: 0.839489	valid_0's binary_logloss: 0.144426	valid_1's auc: 0.831396	valid_1's binary_logloss: 0.153706
    [5]	valid_0's auc: 0.843358	valid_0's binary_logloss: 0.142067	valid_1's auc: 0.833466	valid_1's binary_logloss: 0.151399
    [6]	valid_0's auc: 0.845601	valid_0's binary_logloss: 0.14009	valid_1's auc: 0.833857	valid_1's binary_logloss: 0.149488
    [7]	valid_0's auc: 0.846477	valid_0's binary_logloss: 0.138491	valid_1's auc: 0.833143	valid_1's binary_logloss: 0.148023
    [8]	valid_0's auc: 0.847725	valid_0's binary_logloss: 0.137129	valid_1's auc: 0.833971	valid_1's binary_logloss: 0.146757
    [9]	valid_0's auc: 0.848442	valid_0's binary_logloss: 0.135908	valid_1's auc: 0.835976	valid_1's binary_logloss: 0.145685
    [10]	valid_0's auc: 0.849759	valid_0's binary_logloss: 0.134781	valid_1's auc: 0.836214	valid_1's binary_logloss: 0.144769
    [11]	valid_0's auc: 0.852238	valid_0's binary_logloss: 0.133835	valid_1's auc: 0.837243	valid_1's binary_logloss: 0.143925
    [12]	valid_0's auc: 0.853743	valid_0's binary_logloss: 0.132972	valid_1's auc: 0.836647	valid_1's binary_logloss: 0.143391
    [13]	valid_0's auc: 0.854568	valid_0's binary_logloss: 0.132256	valid_1's auc: 0.837182	valid_1's binary_logloss: 0.142849
    [14]	valid_0's auc: 0.855928	valid_0's binary_logloss: 0.131554	valid_1's auc: 0.835941	valid_1's binary_logloss: 0.142474
    [15]	valid_0's auc: 0.85712	valid_0's binary_logloss: 0.130984	valid_1's auc: 0.834938	valid_1's binary_logloss: 0.142198
    [16]	valid_0's auc: 0.858721	valid_0's binary_logloss: 0.130371	valid_1's auc: 0.83561	valid_1's binary_logloss: 0.141802
    [17]	valid_0's auc: 0.859281	valid_0's binary_logloss: 0.129877	valid_1's auc: 0.835146	valid_1's binary_logloss: 0.141605
    [18]	valid_0's auc: 0.859881	valid_0's binary_logloss: 0.129417	valid_1's auc: 0.835386	valid_1's binary_logloss: 0.14132
    [19]	valid_0's auc: 0.861409	valid_0's binary_logloss: 0.128929	valid_1's auc: 0.834974	valid_1's binary_logloss: 0.141151
    [20]	valid_0's auc: 0.862574	valid_0's binary_logloss: 0.128458	valid_1's auc: 0.834949	valid_1's binary_logloss: 0.140968
    [21]	valid_0's auc: 0.863262	valid_0's binary_logloss: 0.128069	valid_1's auc: 0.834616	valid_1's binary_logloss: 0.14086
    [22]	valid_0's auc: 0.864655	valid_0's binary_logloss: 0.127684	valid_1's auc: 0.834363	valid_1's binary_logloss: 0.140766
    [23]	valid_0's auc: 0.865247	valid_0's binary_logloss: 0.127349	valid_1's auc: 0.834317	valid_1's binary_logloss: 0.140688
    [24]	valid_0's auc: 0.865882	valid_0's binary_logloss: 0.12704	valid_1's auc: 0.833543	valid_1's binary_logloss: 0.14068
    [25]	valid_0's auc: 0.867496	valid_0's binary_logloss: 0.126629	valid_1's auc: 0.834195	valid_1's binary_logloss: 0.140539
    [26]	valid_0's auc: 0.867923	valid_0's binary_logloss: 0.126353	valid_1's auc: 0.834028	valid_1's binary_logloss: 0.140506
    [27]	valid_0's auc: 0.868685	valid_0's binary_logloss: 0.126058	valid_1's auc: 0.834718	valid_1's binary_logloss: 0.140359
    [28]	valid_0's auc: 0.869304	valid_0's binary_logloss: 0.125764	valid_1's auc: 0.834935	valid_1's binary_logloss: 0.140287
    [29]	valid_0's auc: 0.870037	valid_0's binary_logloss: 0.125514	valid_1's auc: 0.834481	valid_1's binary_logloss: 0.140258
    [30]	valid_0's auc: 0.870785	valid_0's binary_logloss: 0.125254	valid_1's auc: 0.834179	valid_1's binary_logloss: 0.140275
    [31]	valid_0's auc: 0.871706	valid_0's binary_logloss: 0.124992	valid_1's auc: 0.834475	valid_1's binary_logloss: 0.140205
    [32]	valid_0's auc: 0.872582	valid_0's binary_logloss: 0.124728	valid_1's auc: 0.834353	valid_1's binary_logloss: 0.140189
    [33]	valid_0's auc: 0.873445	valid_0's binary_logloss: 0.124481	valid_1's auc: 0.834592	valid_1's binary_logloss: 0.140082
    [34]	valid_0's auc: 0.874095	valid_0's binary_logloss: 0.12426	valid_1's auc: 0.83436	valid_1's binary_logloss: 0.140101
    [35]	valid_0's auc: 0.874869	valid_0's binary_logloss: 0.123982	valid_1's auc: 0.834045	valid_1's binary_logloss: 0.140151
    [36]	valid_0's auc: 0.875446	valid_0's binary_logloss: 0.123753	valid_1's auc: 0.834073	valid_1's binary_logloss: 0.140125
    [37]	valid_0's auc: 0.875763	valid_0's binary_logloss: 0.123587	valid_1's auc: 0.833611	valid_1's binary_logloss: 0.140201
    [38]	valid_0's auc: 0.876603	valid_0's binary_logloss: 0.123335	valid_1's auc: 0.833805	valid_1's binary_logloss: 0.140159
    [39]	valid_0's auc: 0.877126	valid_0's binary_logloss: 0.123134	valid_1's auc: 0.834422	valid_1's binary_logloss: 0.140048
    [40]	valid_0's auc: 0.877575	valid_0's binary_logloss: 0.123013	valid_1's auc: 0.834343	valid_1's binary_logloss: 0.140069
    [41]	valid_0's auc: 0.87809	valid_0's binary_logloss: 0.122813	valid_1's auc: 0.834199	valid_1's binary_logloss: 0.140085
    Early stopping, best iteration is:
    [11]	valid_0's auc: 0.852238	valid_0's binary_logloss: 0.133835	valid_1's auc: 0.837243	valid_1's binary_logloss: 0.143925
    [1]	valid_0's auc: 0.821831	valid_0's binary_logloss: 0.156466	valid_1's auc: 0.817525	valid_1's binary_logloss: 0.165186
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.831974	valid_0's binary_logloss: 0.151137	valid_1's auc: 0.82532	valid_1's binary_logloss: 0.159691
    [3]	valid_0's auc: 0.839496	valid_0's binary_logloss: 0.14733	valid_1's auc: 0.831946	valid_1's binary_logloss: 0.156
    [4]	valid_0's auc: 0.843984	valid_0's binary_logloss: 0.144371	valid_1's auc: 0.834064	valid_1's binary_logloss: 0.153082
    [5]	valid_0's auc: 0.845854	valid_0's binary_logloss: 0.142024	valid_1's auc: 0.836918	valid_1's binary_logloss: 0.150735
    [6]	valid_0's auc: 0.848041	valid_0's binary_logloss: 0.140009	valid_1's auc: 0.838831	valid_1's binary_logloss: 0.148771
    [7]	valid_0's auc: 0.849655	valid_0's binary_logloss: 0.138307	valid_1's auc: 0.839111	valid_1's binary_logloss: 0.147373
    [8]	valid_0's auc: 0.85185	valid_0's binary_logloss: 0.136891	valid_1's auc: 0.838955	valid_1's binary_logloss: 0.146094
    [9]	valid_0's auc: 0.853067	valid_0's binary_logloss: 0.135655	valid_1's auc: 0.838081	valid_1's binary_logloss: 0.14516
    [10]	valid_0's auc: 0.853922	valid_0's binary_logloss: 0.134622	valid_1's auc: 0.837333	valid_1's binary_logloss: 0.144318
    [11]	valid_0's auc: 0.854729	valid_0's binary_logloss: 0.133702	valid_1's auc: 0.83725	valid_1's binary_logloss: 0.143512
    [12]	valid_0's auc: 0.856303	valid_0's binary_logloss: 0.132789	valid_1's auc: 0.837602	valid_1's binary_logloss: 0.142833
    [13]	valid_0's auc: 0.857206	valid_0's binary_logloss: 0.132038	valid_1's auc: 0.837364	valid_1's binary_logloss: 0.142245
    [14]	valid_0's auc: 0.858161	valid_0's binary_logloss: 0.131391	valid_1's auc: 0.83777	valid_1's binary_logloss: 0.141759
    [15]	valid_0's auc: 0.858975	valid_0's binary_logloss: 0.130772	valid_1's auc: 0.837831	valid_1's binary_logloss: 0.14139
    [16]	valid_0's auc: 0.859623	valid_0's binary_logloss: 0.130219	valid_1's auc: 0.837953	valid_1's binary_logloss: 0.141016
    [17]	valid_0's auc: 0.860576	valid_0's binary_logloss: 0.129684	valid_1's auc: 0.837985	valid_1's binary_logloss: 0.140713
    [18]	valid_0's auc: 0.861311	valid_0's binary_logloss: 0.129202	valid_1's auc: 0.83796	valid_1's binary_logloss: 0.140452
    [19]	valid_0's auc: 0.862347	valid_0's binary_logloss: 0.128715	valid_1's auc: 0.838506	valid_1's binary_logloss: 0.140189
    [20]	valid_0's auc: 0.86305	valid_0's binary_logloss: 0.128312	valid_1's auc: 0.837702	valid_1's binary_logloss: 0.140094
    [21]	valid_0's auc: 0.863758	valid_0's binary_logloss: 0.127907	valid_1's auc: 0.838127	valid_1's binary_logloss: 0.139858
    [22]	valid_0's auc: 0.864635	valid_0's binary_logloss: 0.127525	valid_1's auc: 0.838331	valid_1's binary_logloss: 0.139696
    [23]	valid_0's auc: 0.865866	valid_0's binary_logloss: 0.127143	valid_1's auc: 0.837841	valid_1's binary_logloss: 0.139625
    [24]	valid_0's auc: 0.867054	valid_0's binary_logloss: 0.126749	valid_1's auc: 0.838187	valid_1's binary_logloss: 0.139526
    [25]	valid_0's auc: 0.867553	valid_0's binary_logloss: 0.126476	valid_1's auc: 0.838308	valid_1's binary_logloss: 0.13949
    [26]	valid_0's auc: 0.868108	valid_0's binary_logloss: 0.126164	valid_1's auc: 0.838035	valid_1's binary_logloss: 0.139426
    [27]	valid_0's auc: 0.869014	valid_0's binary_logloss: 0.125868	valid_1's auc: 0.837545	valid_1's binary_logloss: 0.139445
    [28]	valid_0's auc: 0.869797	valid_0's binary_logloss: 0.12559	valid_1's auc: 0.837894	valid_1's binary_logloss: 0.139419
    [29]	valid_0's auc: 0.870435	valid_0's binary_logloss: 0.1253	valid_1's auc: 0.838103	valid_1's binary_logloss: 0.139321
    [30]	valid_0's auc: 0.87141	valid_0's binary_logloss: 0.125025	valid_1's auc: 0.838164	valid_1's binary_logloss: 0.139275
    [31]	valid_0's auc: 0.872143	valid_0's binary_logloss: 0.124769	valid_1's auc: 0.837843	valid_1's binary_logloss: 0.139285
    [32]	valid_0's auc: 0.872606	valid_0's binary_logloss: 0.124561	valid_1's auc: 0.837662	valid_1's binary_logloss: 0.139274
    [33]	valid_0's auc: 0.873337	valid_0's binary_logloss: 0.124346	valid_1's auc: 0.837661	valid_1's binary_logloss: 0.139284
    [34]	valid_0's auc: 0.873965	valid_0's binary_logloss: 0.124108	valid_1's auc: 0.837639	valid_1's binary_logloss: 0.139263
    [35]	valid_0's auc: 0.87457	valid_0's binary_logloss: 0.123857	valid_1's auc: 0.838159	valid_1's binary_logloss: 0.139137
    [36]	valid_0's auc: 0.874973	valid_0's binary_logloss: 0.123651	valid_1's auc: 0.838114	valid_1's binary_logloss: 0.139148
    [37]	valid_0's auc: 0.875657	valid_0's binary_logloss: 0.123447	valid_1's auc: 0.838519	valid_1's binary_logloss: 0.139109
    Early stopping, best iteration is:
    [7]	valid_0's auc: 0.849655	valid_0's binary_logloss: 0.138307	valid_1's auc: 0.839111	valid_1's binary_logloss: 0.147373
    [1]	valid_0's auc: 0.821427	valid_0's binary_logloss: 0.156592	valid_1's auc: 0.81711	valid_1's binary_logloss: 0.165273
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.827893	valid_0's binary_logloss: 0.151336	valid_1's auc: 0.820533	valid_1's binary_logloss: 0.160243
    [3]	valid_0's auc: 0.83753	valid_0's binary_logloss: 0.147487	valid_1's auc: 0.82841	valid_1's binary_logloss: 0.156547
    [4]	valid_0's auc: 0.84038	valid_0's binary_logloss: 0.144428	valid_1's auc: 0.8313	valid_1's binary_logloss: 0.153575
    [5]	valid_0's auc: 0.842945	valid_0's binary_logloss: 0.142089	valid_1's auc: 0.833579	valid_1's binary_logloss: 0.151354
    [6]	valid_0's auc: 0.843246	valid_0's binary_logloss: 0.140186	valid_1's auc: 0.833781	valid_1's binary_logloss: 0.14953
    [7]	valid_0's auc: 0.844301	valid_0's binary_logloss: 0.138471	valid_1's auc: 0.834317	valid_1's binary_logloss: 0.147954
    [8]	valid_0's auc: 0.846945	valid_0's binary_logloss: 0.137078	valid_1's auc: 0.834895	valid_1's binary_logloss: 0.146786
    [9]	valid_0's auc: 0.849381	valid_0's binary_logloss: 0.135906	valid_1's auc: 0.834922	valid_1's binary_logloss: 0.145762
    [10]	valid_0's auc: 0.850944	valid_0's binary_logloss: 0.134855	valid_1's auc: 0.835441	valid_1's binary_logloss: 0.144958
    [11]	valid_0's auc: 0.852557	valid_0's binary_logloss: 0.133895	valid_1's auc: 0.835103	valid_1's binary_logloss: 0.144293
    [12]	valid_0's auc: 0.854609	valid_0's binary_logloss: 0.133013	valid_1's auc: 0.835686	valid_1's binary_logloss: 0.143793
    [13]	valid_0's auc: 0.855817	valid_0's binary_logloss: 0.132247	valid_1's auc: 0.835296	valid_1's binary_logloss: 0.143302
    [14]	valid_0's auc: 0.857501	valid_0's binary_logloss: 0.131545	valid_1's auc: 0.836432	valid_1's binary_logloss: 0.142761
    [15]	valid_0's auc: 0.858907	valid_0's binary_logloss: 0.130878	valid_1's auc: 0.836329	valid_1's binary_logloss: 0.142383
    [16]	valid_0's auc: 0.859887	valid_0's binary_logloss: 0.130287	valid_1's auc: 0.836611	valid_1's binary_logloss: 0.141883
    [17]	valid_0's auc: 0.860889	valid_0's binary_logloss: 0.129757	valid_1's auc: 0.836848	valid_1's binary_logloss: 0.141535
    [18]	valid_0's auc: 0.861827	valid_0's binary_logloss: 0.129301	valid_1's auc: 0.837106	valid_1's binary_logloss: 0.141257
    [19]	valid_0's auc: 0.862972	valid_0's binary_logloss: 0.128826	valid_1's auc: 0.837185	valid_1's binary_logloss: 0.141043
    [20]	valid_0's auc: 0.864083	valid_0's binary_logloss: 0.128369	valid_1's auc: 0.837509	valid_1's binary_logloss: 0.140794
    [21]	valid_0's auc: 0.864747	valid_0's binary_logloss: 0.127959	valid_1's auc: 0.837888	valid_1's binary_logloss: 0.140626
    [22]	valid_0's auc: 0.865769	valid_0's binary_logloss: 0.127562	valid_1's auc: 0.837811	valid_1's binary_logloss: 0.140487
    [23]	valid_0's auc: 0.866657	valid_0's binary_logloss: 0.127217	valid_1's auc: 0.837884	valid_1's binary_logloss: 0.140328
    [24]	valid_0's auc: 0.867293	valid_0's binary_logloss: 0.126875	valid_1's auc: 0.838481	valid_1's binary_logloss: 0.140215
    [25]	valid_0's auc: 0.867983	valid_0's binary_logloss: 0.126562	valid_1's auc: 0.838239	valid_1's binary_logloss: 0.140124
    [26]	valid_0's auc: 0.868559	valid_0's binary_logloss: 0.126248	valid_1's auc: 0.837903	valid_1's binary_logloss: 0.140092
    [27]	valid_0's auc: 0.869394	valid_0's binary_logloss: 0.125936	valid_1's auc: 0.837493	valid_1's binary_logloss: 0.14006
    [28]	valid_0's auc: 0.87048	valid_0's binary_logloss: 0.125677	valid_1's auc: 0.837623	valid_1's binary_logloss: 0.140007
    [29]	valid_0's auc: 0.87105	valid_0's binary_logloss: 0.125405	valid_1's auc: 0.838216	valid_1's binary_logloss: 0.13986
    [30]	valid_0's auc: 0.871749	valid_0's binary_logloss: 0.125147	valid_1's auc: 0.838898	valid_1's binary_logloss: 0.139742
    [31]	valid_0's auc: 0.87247	valid_0's binary_logloss: 0.124907	valid_1's auc: 0.838959	valid_1's binary_logloss: 0.139727
    [32]	valid_0's auc: 0.87282	valid_0's binary_logloss: 0.124724	valid_1's auc: 0.838675	valid_1's binary_logloss: 0.139761
    [33]	valid_0's auc: 0.874106	valid_0's binary_logloss: 0.124412	valid_1's auc: 0.838893	valid_1's binary_logloss: 0.139687
    [34]	valid_0's auc: 0.874887	valid_0's binary_logloss: 0.124169	valid_1's auc: 0.838801	valid_1's binary_logloss: 0.139672
    [35]	valid_0's auc: 0.875447	valid_0's binary_logloss: 0.123934	valid_1's auc: 0.838835	valid_1's binary_logloss: 0.139667
    [36]	valid_0's auc: 0.87617	valid_0's binary_logloss: 0.123693	valid_1's auc: 0.838505	valid_1's binary_logloss: 0.139699
    [37]	valid_0's auc: 0.876793	valid_0's binary_logloss: 0.12346	valid_1's auc: 0.838104	valid_1's binary_logloss: 0.139783
    [38]	valid_0's auc: 0.877265	valid_0's binary_logloss: 0.123251	valid_1's auc: 0.838267	valid_1's binary_logloss: 0.139787
    [39]	valid_0's auc: 0.877869	valid_0's binary_logloss: 0.123018	valid_1's auc: 0.838004	valid_1's binary_logloss: 0.139806
    [40]	valid_0's auc: 0.878509	valid_0's binary_logloss: 0.122803	valid_1's auc: 0.838086	valid_1's binary_logloss: 0.139745
    [41]	valid_0's auc: 0.879077	valid_0's binary_logloss: 0.122585	valid_1's auc: 0.838538	valid_1's binary_logloss: 0.139694
    [42]	valid_0's auc: 0.879515	valid_0's binary_logloss: 0.122368	valid_1's auc: 0.838647	valid_1's binary_logloss: 0.139655
    [43]	valid_0's auc: 0.879985	valid_0's binary_logloss: 0.122166	valid_1's auc: 0.838495	valid_1's binary_logloss: 0.139653
    [44]	valid_0's auc: 0.88041	valid_0's binary_logloss: 0.121985	valid_1's auc: 0.838221	valid_1's binary_logloss: 0.139755
    [45]	valid_0's auc: 0.880907	valid_0's binary_logloss: 0.121777	valid_1's auc: 0.837981	valid_1's binary_logloss: 0.139769
    [46]	valid_0's auc: 0.881216	valid_0's binary_logloss: 0.121594	valid_1's auc: 0.838471	valid_1's binary_logloss: 0.139693
    [47]	valid_0's auc: 0.881591	valid_0's binary_logloss: 0.121422	valid_1's auc: 0.83861	valid_1's binary_logloss: 0.139687
    [48]	valid_0's auc: 0.881867	valid_0's binary_logloss: 0.121266	valid_1's auc: 0.838593	valid_1's binary_logloss: 0.139682
    [49]	valid_0's auc: 0.882285	valid_0's binary_logloss: 0.121041	valid_1's auc: 0.838317	valid_1's binary_logloss: 0.139741
    [50]	valid_0's auc: 0.882828	valid_0's binary_logloss: 0.120853	valid_1's auc: 0.838244	valid_1's binary_logloss: 0.139759
    [51]	valid_0's auc: 0.883154	valid_0's binary_logloss: 0.120688	valid_1's auc: 0.838222	valid_1's binary_logloss: 0.139803
    [52]	valid_0's auc: 0.883348	valid_0's binary_logloss: 0.120567	valid_1's auc: 0.838064	valid_1's binary_logloss: 0.139824
    [53]	valid_0's auc: 0.883583	valid_0's binary_logloss: 0.120424	valid_1's auc: 0.83788	valid_1's binary_logloss: 0.139844
    [54]	valid_0's auc: 0.884106	valid_0's binary_logloss: 0.120208	valid_1's auc: 0.837625	valid_1's binary_logloss: 0.139886
    [55]	valid_0's auc: 0.884777	valid_0's binary_logloss: 0.120039	valid_1's auc: 0.837585	valid_1's binary_logloss: 0.139902
    [56]	valid_0's auc: 0.88511	valid_0's binary_logloss: 0.11989	valid_1's auc: 0.837646	valid_1's binary_logloss: 0.139926
    [57]	valid_0's auc: 0.885365	valid_0's binary_logloss: 0.11975	valid_1's auc: 0.837639	valid_1's binary_logloss: 0.139934
    [58]	valid_0's auc: 0.885606	valid_0's binary_logloss: 0.119595	valid_1's auc: 0.837726	valid_1's binary_logloss: 0.139938
    [59]	valid_0's auc: 0.885965	valid_0's binary_logloss: 0.119403	valid_1's auc: 0.837558	valid_1's binary_logloss: 0.140007
    [60]	valid_0's auc: 0.886208	valid_0's binary_logloss: 0.119263	valid_1's auc: 0.83744	valid_1's binary_logloss: 0.140079
    [61]	valid_0's auc: 0.886458	valid_0's binary_logloss: 0.119118	valid_1's auc: 0.837349	valid_1's binary_logloss: 0.140059
    Early stopping, best iteration is:
    [31]	valid_0's auc: 0.87247	valid_0's binary_logloss: 0.124907	valid_1's auc: 0.838959	valid_1's binary_logloss: 0.139727
    [1]	valid_0's auc: 0.835412	valid_0's binary_logloss: 0.155721	valid_1's auc: 0.81973	valid_1's binary_logloss: 0.164844
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.841188	valid_0's binary_logloss: 0.150354	valid_1's auc: 0.823402	valid_1's binary_logloss: 0.16006
    [3]	valid_0's auc: 0.846758	valid_0's binary_logloss: 0.146288	valid_1's auc: 0.824811	valid_1's binary_logloss: 0.15621
    [4]	valid_0's auc: 0.850398	valid_0's binary_logloss: 0.143088	valid_1's auc: 0.830278	valid_1's binary_logloss: 0.153352
    [5]	valid_0's auc: 0.853086	valid_0's binary_logloss: 0.140514	valid_1's auc: 0.833574	valid_1's binary_logloss: 0.151071
    [6]	valid_0's auc: 0.855915	valid_0's binary_logloss: 0.138329	valid_1's auc: 0.834881	valid_1's binary_logloss: 0.149277
    [7]	valid_0's auc: 0.858115	valid_0's binary_logloss: 0.136481	valid_1's auc: 0.833603	valid_1's binary_logloss: 0.14786
    [8]	valid_0's auc: 0.859479	valid_0's binary_logloss: 0.134947	valid_1's auc: 0.834093	valid_1's binary_logloss: 0.146607
    [9]	valid_0's auc: 0.86143	valid_0's binary_logloss: 0.133519	valid_1's auc: 0.833898	valid_1's binary_logloss: 0.14559
    [10]	valid_0's auc: 0.862964	valid_0's binary_logloss: 0.132331	valid_1's auc: 0.835026	valid_1's binary_logloss: 0.144789
    [11]	valid_0's auc: 0.864277	valid_0's binary_logloss: 0.13126	valid_1's auc: 0.834957	valid_1's binary_logloss: 0.144152
    [12]	valid_0's auc: 0.865572	valid_0's binary_logloss: 0.130304	valid_1's auc: 0.833693	valid_1's binary_logloss: 0.143697
    [13]	valid_0's auc: 0.867519	valid_0's binary_logloss: 0.129385	valid_1's auc: 0.833158	valid_1's binary_logloss: 0.143184
    [14]	valid_0's auc: 0.869354	valid_0's binary_logloss: 0.128524	valid_1's auc: 0.833598	valid_1's binary_logloss: 0.142668
    [15]	valid_0's auc: 0.870553	valid_0's binary_logloss: 0.127746	valid_1's auc: 0.833467	valid_1's binary_logloss: 0.142302
    [16]	valid_0's auc: 0.871816	valid_0's binary_logloss: 0.126943	valid_1's auc: 0.83329	valid_1's binary_logloss: 0.142022
    [17]	valid_0's auc: 0.872964	valid_0's binary_logloss: 0.126266	valid_1's auc: 0.83279	valid_1's binary_logloss: 0.141891
    [18]	valid_0's auc: 0.874047	valid_0's binary_logloss: 0.125646	valid_1's auc: 0.831917	valid_1's binary_logloss: 0.141748
    [19]	valid_0's auc: 0.875336	valid_0's binary_logloss: 0.125072	valid_1's auc: 0.831274	valid_1's binary_logloss: 0.141658
    [20]	valid_0's auc: 0.876959	valid_0's binary_logloss: 0.124484	valid_1's auc: 0.831275	valid_1's binary_logloss: 0.141511
    [21]	valid_0's auc: 0.878049	valid_0's binary_logloss: 0.123928	valid_1's auc: 0.830813	valid_1's binary_logloss: 0.141459
    [22]	valid_0's auc: 0.878905	valid_0's binary_logloss: 0.123447	valid_1's auc: 0.83012	valid_1's binary_logloss: 0.141449
    [23]	valid_0's auc: 0.879827	valid_0's binary_logloss: 0.12295	valid_1's auc: 0.829554	valid_1's binary_logloss: 0.141492
    [24]	valid_0's auc: 0.880692	valid_0's binary_logloss: 0.122479	valid_1's auc: 0.829256	valid_1's binary_logloss: 0.141487
    [25]	valid_0's auc: 0.881715	valid_0's binary_logloss: 0.121994	valid_1's auc: 0.829326	valid_1's binary_logloss: 0.141362
    [26]	valid_0's auc: 0.883014	valid_0's binary_logloss: 0.121527	valid_1's auc: 0.829553	valid_1's binary_logloss: 0.14132
    [27]	valid_0's auc: 0.884245	valid_0's binary_logloss: 0.121024	valid_1's auc: 0.829624	valid_1's binary_logloss: 0.14127
    [28]	valid_0's auc: 0.885238	valid_0's binary_logloss: 0.12058	valid_1's auc: 0.829417	valid_1's binary_logloss: 0.141237
    [29]	valid_0's auc: 0.88602	valid_0's binary_logloss: 0.120198	valid_1's auc: 0.82917	valid_1's binary_logloss: 0.141201
    [30]	valid_0's auc: 0.88684	valid_0's binary_logloss: 0.119831	valid_1's auc: 0.82962	valid_1's binary_logloss: 0.141121
    [31]	valid_0's auc: 0.887965	valid_0's binary_logloss: 0.119437	valid_1's auc: 0.83035	valid_1's binary_logloss: 0.14101
    [32]	valid_0's auc: 0.88868	valid_0's binary_logloss: 0.119086	valid_1's auc: 0.82975	valid_1's binary_logloss: 0.141093
    [33]	valid_0's auc: 0.889895	valid_0's binary_logloss: 0.118649	valid_1's auc: 0.829977	valid_1's binary_logloss: 0.141037
    [34]	valid_0's auc: 0.890626	valid_0's binary_logloss: 0.118328	valid_1's auc: 0.829368	valid_1's binary_logloss: 0.141161
    [35]	valid_0's auc: 0.89116	valid_0's binary_logloss: 0.11806	valid_1's auc: 0.829262	valid_1's binary_logloss: 0.141183
    [36]	valid_0's auc: 0.891999	valid_0's binary_logloss: 0.11775	valid_1's auc: 0.828947	valid_1's binary_logloss: 0.14129
    [37]	valid_0's auc: 0.892306	valid_0's binary_logloss: 0.117477	valid_1's auc: 0.828544	valid_1's binary_logloss: 0.141389
    [38]	valid_0's auc: 0.892937	valid_0's binary_logloss: 0.117192	valid_1's auc: 0.827983	valid_1's binary_logloss: 0.141516
    [39]	valid_0's auc: 0.893563	valid_0's binary_logloss: 0.116869	valid_1's auc: 0.828068	valid_1's binary_logloss: 0.141517
    [40]	valid_0's auc: 0.893942	valid_0's binary_logloss: 0.11662	valid_1's auc: 0.827852	valid_1's binary_logloss: 0.141621
    Early stopping, best iteration is:
    [10]	valid_0's auc: 0.862964	valid_0's binary_logloss: 0.132331	valid_1's auc: 0.835026	valid_1's binary_logloss: 0.144789
    [1]	valid_0's auc: 0.830474	valid_0's binary_logloss: 0.155928	valid_1's auc: 0.817343	valid_1's binary_logloss: 0.164928
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.842931	valid_0's binary_logloss: 0.1503	valid_1's auc: 0.82699	valid_1's binary_logloss: 0.15948
    [3]	valid_0's auc: 0.850877	valid_0's binary_logloss: 0.14631	valid_1's auc: 0.832212	valid_1's binary_logloss: 0.155775
    [4]	valid_0's auc: 0.854431	valid_0's binary_logloss: 0.143104	valid_1's auc: 0.83392	valid_1's binary_logloss: 0.152698
    [5]	valid_0's auc: 0.85663	valid_0's binary_logloss: 0.140582	valid_1's auc: 0.835094	valid_1's binary_logloss: 0.150349
    [6]	valid_0's auc: 0.859142	valid_0's binary_logloss: 0.138289	valid_1's auc: 0.836166	valid_1's binary_logloss: 0.148424
    [7]	valid_0's auc: 0.861364	valid_0's binary_logloss: 0.136413	valid_1's auc: 0.837184	valid_1's binary_logloss: 0.146912
    [8]	valid_0's auc: 0.862199	valid_0's binary_logloss: 0.134841	valid_1's auc: 0.837545	valid_1's binary_logloss: 0.145726
    [9]	valid_0's auc: 0.864095	valid_0's binary_logloss: 0.133364	valid_1's auc: 0.837242	valid_1's binary_logloss: 0.144736
    [10]	valid_0's auc: 0.866024	valid_0's binary_logloss: 0.132096	valid_1's auc: 0.837719	valid_1's binary_logloss: 0.143766
    [11]	valid_0's auc: 0.867454	valid_0's binary_logloss: 0.131002	valid_1's auc: 0.837865	valid_1's binary_logloss: 0.143009
    [12]	valid_0's auc: 0.868329	valid_0's binary_logloss: 0.130024	valid_1's auc: 0.837259	valid_1's binary_logloss: 0.14244
    [13]	valid_0's auc: 0.869137	valid_0's binary_logloss: 0.129145	valid_1's auc: 0.837689	valid_1's binary_logloss: 0.141896
    [14]	valid_0's auc: 0.870957	valid_0's binary_logloss: 0.128226	valid_1's auc: 0.838226	valid_1's binary_logloss: 0.141392
    [15]	valid_0's auc: 0.872273	valid_0's binary_logloss: 0.12745	valid_1's auc: 0.837906	valid_1's binary_logloss: 0.141019
    [16]	valid_0's auc: 0.873243	valid_0's binary_logloss: 0.12672	valid_1's auc: 0.837761	valid_1's binary_logloss: 0.140677
    [17]	valid_0's auc: 0.874251	valid_0's binary_logloss: 0.126044	valid_1's auc: 0.83701	valid_1's binary_logloss: 0.140582
    [18]	valid_0's auc: 0.875622	valid_0's binary_logloss: 0.125387	valid_1's auc: 0.836179	valid_1's binary_logloss: 0.140485
    [19]	valid_0's auc: 0.877031	valid_0's binary_logloss: 0.124759	valid_1's auc: 0.836188	valid_1's binary_logloss: 0.14029
    [20]	valid_0's auc: 0.878046	valid_0's binary_logloss: 0.124156	valid_1's auc: 0.836531	valid_1's binary_logloss: 0.140133
    [21]	valid_0's auc: 0.879478	valid_0's binary_logloss: 0.123507	valid_1's auc: 0.837068	valid_1's binary_logloss: 0.13995
    [22]	valid_0's auc: 0.880423	valid_0's binary_logloss: 0.123029	valid_1's auc: 0.836817	valid_1's binary_logloss: 0.139912
    [23]	valid_0's auc: 0.881684	valid_0's binary_logloss: 0.122492	valid_1's auc: 0.836983	valid_1's binary_logloss: 0.139762
    [24]	valid_0's auc: 0.882873	valid_0's binary_logloss: 0.121986	valid_1's auc: 0.837319	valid_1's binary_logloss: 0.139659
    [25]	valid_0's auc: 0.883597	valid_0's binary_logloss: 0.121566	valid_1's auc: 0.837154	valid_1's binary_logloss: 0.139623
    [26]	valid_0's auc: 0.884814	valid_0's binary_logloss: 0.121104	valid_1's auc: 0.836302	valid_1's binary_logloss: 0.139668
    [27]	valid_0's auc: 0.886026	valid_0's binary_logloss: 0.120635	valid_1's auc: 0.836521	valid_1's binary_logloss: 0.139601
    [28]	valid_0's auc: 0.887071	valid_0's binary_logloss: 0.120222	valid_1's auc: 0.836646	valid_1's binary_logloss: 0.139557
    [29]	valid_0's auc: 0.887946	valid_0's binary_logloss: 0.119804	valid_1's auc: 0.836735	valid_1's binary_logloss: 0.139518
    [30]	valid_0's auc: 0.88898	valid_0's binary_logloss: 0.119416	valid_1's auc: 0.836858	valid_1's binary_logloss: 0.139499
    [31]	valid_0's auc: 0.889792	valid_0's binary_logloss: 0.119058	valid_1's auc: 0.836917	valid_1's binary_logloss: 0.139463
    [32]	valid_0's auc: 0.890876	valid_0's binary_logloss: 0.118631	valid_1's auc: 0.836346	valid_1's binary_logloss: 0.139532
    [33]	valid_0's auc: 0.891629	valid_0's binary_logloss: 0.118259	valid_1's auc: 0.836206	valid_1's binary_logloss: 0.139603
    [34]	valid_0's auc: 0.892446	valid_0's binary_logloss: 0.117893	valid_1's auc: 0.836005	valid_1's binary_logloss: 0.139603
    [35]	valid_0's auc: 0.893407	valid_0's binary_logloss: 0.11752	valid_1's auc: 0.8361	valid_1's binary_logloss: 0.139574
    [36]	valid_0's auc: 0.893836	valid_0's binary_logloss: 0.117247	valid_1's auc: 0.836147	valid_1's binary_logloss: 0.139608
    [37]	valid_0's auc: 0.894774	valid_0's binary_logloss: 0.116913	valid_1's auc: 0.836601	valid_1's binary_logloss: 0.139569
    [38]	valid_0's auc: 0.895494	valid_0's binary_logloss: 0.116611	valid_1's auc: 0.836232	valid_1's binary_logloss: 0.139645
    [39]	valid_0's auc: 0.896102	valid_0's binary_logloss: 0.116275	valid_1's auc: 0.836415	valid_1's binary_logloss: 0.139653
    [40]	valid_0's auc: 0.896715	valid_0's binary_logloss: 0.115934	valid_1's auc: 0.836463	valid_1's binary_logloss: 0.139671
    [41]	valid_0's auc: 0.897232	valid_0's binary_logloss: 0.115612	valid_1's auc: 0.836223	valid_1's binary_logloss: 0.139762
    [42]	valid_0's auc: 0.897875	valid_0's binary_logloss: 0.11528	valid_1's auc: 0.836151	valid_1's binary_logloss: 0.139777
    [43]	valid_0's auc: 0.898493	valid_0's binary_logloss: 0.114999	valid_1's auc: 0.836216	valid_1's binary_logloss: 0.139761
    [44]	valid_0's auc: 0.899179	valid_0's binary_logloss: 0.114703	valid_1's auc: 0.836328	valid_1's binary_logloss: 0.139755
    Early stopping, best iteration is:
    [14]	valid_0's auc: 0.870957	valid_0's binary_logloss: 0.128226	valid_1's auc: 0.838226	valid_1's binary_logloss: 0.141392
    [1]	valid_0's auc: 0.834724	valid_0's binary_logloss: 0.15607	valid_1's auc: 0.822983	valid_1's binary_logloss: 0.165104
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.842835	valid_0's binary_logloss: 0.150494	valid_1's auc: 0.830472	valid_1's binary_logloss: 0.159671
    [3]	valid_0's auc: 0.847187	valid_0's binary_logloss: 0.146306	valid_1's auc: 0.830873	valid_1's binary_logloss: 0.155985
    [4]	valid_0's auc: 0.850394	valid_0's binary_logloss: 0.143088	valid_1's auc: 0.830975	valid_1's binary_logloss: 0.15321
    [5]	valid_0's auc: 0.853379	valid_0's binary_logloss: 0.140508	valid_1's auc: 0.832135	valid_1's binary_logloss: 0.150854
    [6]	valid_0's auc: 0.855463	valid_0's binary_logloss: 0.138297	valid_1's auc: 0.833116	valid_1's binary_logloss: 0.149013
    [7]	valid_0's auc: 0.856723	valid_0's binary_logloss: 0.136504	valid_1's auc: 0.833811	valid_1's binary_logloss: 0.147577
    [8]	valid_0's auc: 0.858076	valid_0's binary_logloss: 0.13495	valid_1's auc: 0.835315	valid_1's binary_logloss: 0.146273
    [9]	valid_0's auc: 0.861024	valid_0's binary_logloss: 0.133583	valid_1's auc: 0.835042	valid_1's binary_logloss: 0.145374
    [10]	valid_0's auc: 0.862281	valid_0's binary_logloss: 0.132357	valid_1's auc: 0.834154	valid_1's binary_logloss: 0.144649
    [11]	valid_0's auc: 0.864612	valid_0's binary_logloss: 0.131283	valid_1's auc: 0.834587	valid_1's binary_logloss: 0.143941
    [12]	valid_0's auc: 0.866377	valid_0's binary_logloss: 0.130299	valid_1's auc: 0.834242	valid_1's binary_logloss: 0.143366
    [13]	valid_0's auc: 0.868343	valid_0's binary_logloss: 0.129417	valid_1's auc: 0.833273	valid_1's binary_logloss: 0.142976
    [14]	valid_0's auc: 0.86957	valid_0's binary_logloss: 0.128593	valid_1's auc: 0.833783	valid_1's binary_logloss: 0.142567
    [15]	valid_0's auc: 0.871109	valid_0's binary_logloss: 0.127759	valid_1's auc: 0.834057	valid_1's binary_logloss: 0.142234
    [16]	valid_0's auc: 0.872893	valid_0's binary_logloss: 0.126996	valid_1's auc: 0.835329	valid_1's binary_logloss: 0.141809
    [17]	valid_0's auc: 0.874236	valid_0's binary_logloss: 0.12631	valid_1's auc: 0.834985	valid_1's binary_logloss: 0.141613
    [18]	valid_0's auc: 0.875324	valid_0's binary_logloss: 0.125725	valid_1's auc: 0.834942	valid_1's binary_logloss: 0.141363
    [19]	valid_0's auc: 0.876659	valid_0's binary_logloss: 0.125068	valid_1's auc: 0.835024	valid_1's binary_logloss: 0.141162
    [20]	valid_0's auc: 0.877885	valid_0's binary_logloss: 0.124484	valid_1's auc: 0.835893	valid_1's binary_logloss: 0.140933
    [21]	valid_0's auc: 0.879121	valid_0's binary_logloss: 0.12391	valid_1's auc: 0.837029	valid_1's binary_logloss: 0.140651
    [22]	valid_0's auc: 0.880116	valid_0's binary_logloss: 0.123339	valid_1's auc: 0.837366	valid_1's binary_logloss: 0.140547
    [23]	valid_0's auc: 0.881224	valid_0's binary_logloss: 0.12282	valid_1's auc: 0.837357	valid_1's binary_logloss: 0.140445
    [24]	valid_0's auc: 0.882014	valid_0's binary_logloss: 0.122386	valid_1's auc: 0.837343	valid_1's binary_logloss: 0.140371
    [25]	valid_0's auc: 0.88318	valid_0's binary_logloss: 0.121861	valid_1's auc: 0.83723	valid_1's binary_logloss: 0.140313
    [26]	valid_0's auc: 0.884008	valid_0's binary_logloss: 0.121441	valid_1's auc: 0.837761	valid_1's binary_logloss: 0.140173
    [27]	valid_0's auc: 0.884676	valid_0's binary_logloss: 0.121001	valid_1's auc: 0.838046	valid_1's binary_logloss: 0.140086
    [28]	valid_0's auc: 0.885524	valid_0's binary_logloss: 0.120598	valid_1's auc: 0.838029	valid_1's binary_logloss: 0.140051
    [29]	valid_0's auc: 0.886461	valid_0's binary_logloss: 0.120157	valid_1's auc: 0.837775	valid_1's binary_logloss: 0.140057
    [30]	valid_0's auc: 0.887053	valid_0's binary_logloss: 0.119807	valid_1's auc: 0.837472	valid_1's binary_logloss: 0.140111
    [31]	valid_0's auc: 0.888177	valid_0's binary_logloss: 0.119425	valid_1's auc: 0.837575	valid_1's binary_logloss: 0.140093
    [32]	valid_0's auc: 0.889072	valid_0's binary_logloss: 0.119055	valid_1's auc: 0.837158	valid_1's binary_logloss: 0.140195
    [33]	valid_0's auc: 0.889782	valid_0's binary_logloss: 0.118676	valid_1's auc: 0.837296	valid_1's binary_logloss: 0.140221
    [34]	valid_0's auc: 0.890876	valid_0's binary_logloss: 0.118304	valid_1's auc: 0.837481	valid_1's binary_logloss: 0.140165
    [35]	valid_0's auc: 0.891448	valid_0's binary_logloss: 0.11798	valid_1's auc: 0.837953	valid_1's binary_logloss: 0.140085
    [36]	valid_0's auc: 0.892165	valid_0's binary_logloss: 0.11764	valid_1's auc: 0.837794	valid_1's binary_logloss: 0.140112
    [37]	valid_0's auc: 0.892798	valid_0's binary_logloss: 0.117321	valid_1's auc: 0.837291	valid_1's binary_logloss: 0.140221
    [38]	valid_0's auc: 0.893318	valid_0's binary_logloss: 0.117028	valid_1's auc: 0.837278	valid_1's binary_logloss: 0.140221
    [39]	valid_0's auc: 0.894018	valid_0's binary_logloss: 0.116742	valid_1's auc: 0.83724	valid_1's binary_logloss: 0.140232
    [40]	valid_0's auc: 0.894781	valid_0's binary_logloss: 0.116373	valid_1's auc: 0.836901	valid_1's binary_logloss: 0.140328
    [41]	valid_0's auc: 0.895222	valid_0's binary_logloss: 0.116075	valid_1's auc: 0.836655	valid_1's binary_logloss: 0.140422
    [42]	valid_0's auc: 0.895842	valid_0's binary_logloss: 0.115755	valid_1's auc: 0.836383	valid_1's binary_logloss: 0.140503
    [43]	valid_0's auc: 0.896389	valid_0's binary_logloss: 0.115503	valid_1's auc: 0.836348	valid_1's binary_logloss: 0.140505
    [44]	valid_0's auc: 0.896843	valid_0's binary_logloss: 0.115204	valid_1's auc: 0.836521	valid_1's binary_logloss: 0.140518
    [45]	valid_0's auc: 0.897272	valid_0's binary_logloss: 0.114886	valid_1's auc: 0.836311	valid_1's binary_logloss: 0.140581
    [46]	valid_0's auc: 0.898034	valid_0's binary_logloss: 0.114544	valid_1's auc: 0.835871	valid_1's binary_logloss: 0.140663
    [47]	valid_0's auc: 0.898562	valid_0's binary_logloss: 0.114262	valid_1's auc: 0.835926	valid_1's binary_logloss: 0.140642
    [48]	valid_0's auc: 0.898919	valid_0's binary_logloss: 0.114006	valid_1's auc: 0.835849	valid_1's binary_logloss: 0.140687
    [49]	valid_0's auc: 0.899111	valid_0's binary_logloss: 0.113791	valid_1's auc: 0.835874	valid_1's binary_logloss: 0.140728
    [50]	valid_0's auc: 0.89987	valid_0's binary_logloss: 0.113543	valid_1's auc: 0.835915	valid_1's binary_logloss: 0.14075
    [51]	valid_0's auc: 0.90004	valid_0's binary_logloss: 0.113342	valid_1's auc: 0.835947	valid_1's binary_logloss: 0.140748
    [52]	valid_0's auc: 0.900405	valid_0's binary_logloss: 0.113087	valid_1's auc: 0.836011	valid_1's binary_logloss: 0.140767
    [53]	valid_0's auc: 0.900828	valid_0's binary_logloss: 0.112831	valid_1's auc: 0.836259	valid_1's binary_logloss: 0.140771
    [54]	valid_0's auc: 0.901597	valid_0's binary_logloss: 0.112604	valid_1's auc: 0.836296	valid_1's binary_logloss: 0.14078
    [55]	valid_0's auc: 0.901645	valid_0's binary_logloss: 0.112429	valid_1's auc: 0.836095	valid_1's binary_logloss: 0.140822
    [56]	valid_0's auc: 0.902162	valid_0's binary_logloss: 0.112169	valid_1's auc: 0.835965	valid_1's binary_logloss: 0.14086
    [57]	valid_0's auc: 0.902422	valid_0's binary_logloss: 0.111944	valid_1's auc: 0.835493	valid_1's binary_logloss: 0.140993
    Early stopping, best iteration is:
    [27]	valid_0's auc: 0.884676	valid_0's binary_logloss: 0.121001	valid_1's auc: 0.838046	valid_1's binary_logloss: 0.140086
    [1]	valid_0's auc: 0.835412	valid_0's binary_logloss: 0.155721	valid_1's auc: 0.81973	valid_1's binary_logloss: 0.164844
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.841188	valid_0's binary_logloss: 0.150354	valid_1's auc: 0.823402	valid_1's binary_logloss: 0.16006
    [3]	valid_0's auc: 0.846758	valid_0's binary_logloss: 0.146288	valid_1's auc: 0.824811	valid_1's binary_logloss: 0.15621
    [4]	valid_0's auc: 0.850398	valid_0's binary_logloss: 0.143088	valid_1's auc: 0.830278	valid_1's binary_logloss: 0.153352
    [5]	valid_0's auc: 0.853086	valid_0's binary_logloss: 0.140514	valid_1's auc: 0.833574	valid_1's binary_logloss: 0.151071
    [6]	valid_0's auc: 0.855915	valid_0's binary_logloss: 0.138329	valid_1's auc: 0.834881	valid_1's binary_logloss: 0.149277
    [7]	valid_0's auc: 0.858115	valid_0's binary_logloss: 0.136481	valid_1's auc: 0.833603	valid_1's binary_logloss: 0.14786
    [8]	valid_0's auc: 0.859479	valid_0's binary_logloss: 0.134947	valid_1's auc: 0.834093	valid_1's binary_logloss: 0.146607
    [9]	valid_0's auc: 0.86143	valid_0's binary_logloss: 0.133519	valid_1's auc: 0.833898	valid_1's binary_logloss: 0.14559
    [10]	valid_0's auc: 0.862964	valid_0's binary_logloss: 0.132331	valid_1's auc: 0.835026	valid_1's binary_logloss: 0.144789
    [11]	valid_0's auc: 0.864277	valid_0's binary_logloss: 0.13126	valid_1's auc: 0.834957	valid_1's binary_logloss: 0.144152
    [12]	valid_0's auc: 0.865572	valid_0's binary_logloss: 0.130304	valid_1's auc: 0.833693	valid_1's binary_logloss: 0.143697
    [13]	valid_0's auc: 0.867519	valid_0's binary_logloss: 0.129385	valid_1's auc: 0.833158	valid_1's binary_logloss: 0.143184
    [14]	valid_0's auc: 0.869354	valid_0's binary_logloss: 0.128524	valid_1's auc: 0.833598	valid_1's binary_logloss: 0.142668
    [15]	valid_0's auc: 0.870553	valid_0's binary_logloss: 0.127746	valid_1's auc: 0.833467	valid_1's binary_logloss: 0.142302
    [16]	valid_0's auc: 0.871816	valid_0's binary_logloss: 0.126943	valid_1's auc: 0.83329	valid_1's binary_logloss: 0.142022
    [17]	valid_0's auc: 0.872964	valid_0's binary_logloss: 0.126266	valid_1's auc: 0.83279	valid_1's binary_logloss: 0.141891
    [18]	valid_0's auc: 0.874047	valid_0's binary_logloss: 0.125646	valid_1's auc: 0.831917	valid_1's binary_logloss: 0.141748
    [19]	valid_0's auc: 0.875336	valid_0's binary_logloss: 0.125072	valid_1's auc: 0.831274	valid_1's binary_logloss: 0.141658
    [20]	valid_0's auc: 0.876959	valid_0's binary_logloss: 0.124484	valid_1's auc: 0.831275	valid_1's binary_logloss: 0.141511
    [21]	valid_0's auc: 0.878049	valid_0's binary_logloss: 0.123928	valid_1's auc: 0.830813	valid_1's binary_logloss: 0.141459
    [22]	valid_0's auc: 0.878905	valid_0's binary_logloss: 0.123447	valid_1's auc: 0.83012	valid_1's binary_logloss: 0.141449
    [23]	valid_0's auc: 0.879827	valid_0's binary_logloss: 0.12295	valid_1's auc: 0.829554	valid_1's binary_logloss: 0.141492
    [24]	valid_0's auc: 0.880692	valid_0's binary_logloss: 0.122479	valid_1's auc: 0.829256	valid_1's binary_logloss: 0.141487
    [25]	valid_0's auc: 0.881715	valid_0's binary_logloss: 0.121994	valid_1's auc: 0.829326	valid_1's binary_logloss: 0.141362
    [26]	valid_0's auc: 0.883014	valid_0's binary_logloss: 0.121527	valid_1's auc: 0.829553	valid_1's binary_logloss: 0.14132
    [27]	valid_0's auc: 0.884245	valid_0's binary_logloss: 0.121024	valid_1's auc: 0.829624	valid_1's binary_logloss: 0.14127
    [28]	valid_0's auc: 0.885238	valid_0's binary_logloss: 0.12058	valid_1's auc: 0.829417	valid_1's binary_logloss: 0.141237
    [29]	valid_0's auc: 0.88602	valid_0's binary_logloss: 0.120198	valid_1's auc: 0.82917	valid_1's binary_logloss: 0.141201
    [30]	valid_0's auc: 0.88684	valid_0's binary_logloss: 0.119831	valid_1's auc: 0.82962	valid_1's binary_logloss: 0.141121
    [31]	valid_0's auc: 0.887965	valid_0's binary_logloss: 0.119437	valid_1's auc: 0.83035	valid_1's binary_logloss: 0.14101
    [32]	valid_0's auc: 0.88868	valid_0's binary_logloss: 0.119086	valid_1's auc: 0.82975	valid_1's binary_logloss: 0.141093
    [33]	valid_0's auc: 0.889895	valid_0's binary_logloss: 0.118649	valid_1's auc: 0.829977	valid_1's binary_logloss: 0.141037
    [34]	valid_0's auc: 0.890626	valid_0's binary_logloss: 0.118328	valid_1's auc: 0.829368	valid_1's binary_logloss: 0.141161
    [35]	valid_0's auc: 0.89116	valid_0's binary_logloss: 0.11806	valid_1's auc: 0.829262	valid_1's binary_logloss: 0.141183
    [36]	valid_0's auc: 0.891999	valid_0's binary_logloss: 0.11775	valid_1's auc: 0.828947	valid_1's binary_logloss: 0.14129
    [37]	valid_0's auc: 0.892306	valid_0's binary_logloss: 0.117477	valid_1's auc: 0.828544	valid_1's binary_logloss: 0.141389
    [38]	valid_0's auc: 0.892937	valid_0's binary_logloss: 0.117192	valid_1's auc: 0.827983	valid_1's binary_logloss: 0.141516
    [39]	valid_0's auc: 0.893563	valid_0's binary_logloss: 0.116869	valid_1's auc: 0.828068	valid_1's binary_logloss: 0.141517
    [40]	valid_0's auc: 0.893942	valid_0's binary_logloss: 0.11662	valid_1's auc: 0.827852	valid_1's binary_logloss: 0.141621
    Early stopping, best iteration is:
    [10]	valid_0's auc: 0.862964	valid_0's binary_logloss: 0.132331	valid_1's auc: 0.835026	valid_1's binary_logloss: 0.144789
    [1]	valid_0's auc: 0.830474	valid_0's binary_logloss: 0.155928	valid_1's auc: 0.817343	valid_1's binary_logloss: 0.164928
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.842931	valid_0's binary_logloss: 0.1503	valid_1's auc: 0.82699	valid_1's binary_logloss: 0.15948
    [3]	valid_0's auc: 0.850877	valid_0's binary_logloss: 0.14631	valid_1's auc: 0.832212	valid_1's binary_logloss: 0.155775
    [4]	valid_0's auc: 0.854431	valid_0's binary_logloss: 0.143104	valid_1's auc: 0.83392	valid_1's binary_logloss: 0.152698
    [5]	valid_0's auc: 0.85663	valid_0's binary_logloss: 0.140582	valid_1's auc: 0.835094	valid_1's binary_logloss: 0.150349
    [6]	valid_0's auc: 0.859142	valid_0's binary_logloss: 0.138289	valid_1's auc: 0.836166	valid_1's binary_logloss: 0.148424
    [7]	valid_0's auc: 0.861364	valid_0's binary_logloss: 0.136413	valid_1's auc: 0.837184	valid_1's binary_logloss: 0.146912
    [8]	valid_0's auc: 0.862199	valid_0's binary_logloss: 0.134841	valid_1's auc: 0.837545	valid_1's binary_logloss: 0.145726
    [9]	valid_0's auc: 0.864095	valid_0's binary_logloss: 0.133364	valid_1's auc: 0.837242	valid_1's binary_logloss: 0.144736
    [10]	valid_0's auc: 0.866024	valid_0's binary_logloss: 0.132096	valid_1's auc: 0.837719	valid_1's binary_logloss: 0.143766
    [11]	valid_0's auc: 0.867454	valid_0's binary_logloss: 0.131002	valid_1's auc: 0.837865	valid_1's binary_logloss: 0.143009
    [12]	valid_0's auc: 0.868329	valid_0's binary_logloss: 0.130024	valid_1's auc: 0.837259	valid_1's binary_logloss: 0.14244
    [13]	valid_0's auc: 0.869137	valid_0's binary_logloss: 0.129145	valid_1's auc: 0.837689	valid_1's binary_logloss: 0.141896
    [14]	valid_0's auc: 0.870957	valid_0's binary_logloss: 0.128226	valid_1's auc: 0.838226	valid_1's binary_logloss: 0.141392
    [15]	valid_0's auc: 0.872273	valid_0's binary_logloss: 0.12745	valid_1's auc: 0.837906	valid_1's binary_logloss: 0.141019
    [16]	valid_0's auc: 0.873243	valid_0's binary_logloss: 0.12672	valid_1's auc: 0.837761	valid_1's binary_logloss: 0.140677
    [17]	valid_0's auc: 0.874251	valid_0's binary_logloss: 0.126044	valid_1's auc: 0.83701	valid_1's binary_logloss: 0.140582
    [18]	valid_0's auc: 0.875622	valid_0's binary_logloss: 0.125387	valid_1's auc: 0.836179	valid_1's binary_logloss: 0.140485
    [19]	valid_0's auc: 0.877031	valid_0's binary_logloss: 0.124759	valid_1's auc: 0.836188	valid_1's binary_logloss: 0.14029
    [20]	valid_0's auc: 0.878046	valid_0's binary_logloss: 0.124156	valid_1's auc: 0.836531	valid_1's binary_logloss: 0.140133
    [21]	valid_0's auc: 0.879478	valid_0's binary_logloss: 0.123507	valid_1's auc: 0.837068	valid_1's binary_logloss: 0.13995
    [22]	valid_0's auc: 0.880423	valid_0's binary_logloss: 0.123029	valid_1's auc: 0.836817	valid_1's binary_logloss: 0.139912
    [23]	valid_0's auc: 0.881684	valid_0's binary_logloss: 0.122492	valid_1's auc: 0.836983	valid_1's binary_logloss: 0.139762
    [24]	valid_0's auc: 0.882873	valid_0's binary_logloss: 0.121986	valid_1's auc: 0.837319	valid_1's binary_logloss: 0.139659
    [25]	valid_0's auc: 0.883597	valid_0's binary_logloss: 0.121566	valid_1's auc: 0.837154	valid_1's binary_logloss: 0.139623
    [26]	valid_0's auc: 0.884814	valid_0's binary_logloss: 0.121104	valid_1's auc: 0.836302	valid_1's binary_logloss: 0.139668
    [27]	valid_0's auc: 0.886026	valid_0's binary_logloss: 0.120635	valid_1's auc: 0.836521	valid_1's binary_logloss: 0.139601
    [28]	valid_0's auc: 0.887071	valid_0's binary_logloss: 0.120222	valid_1's auc: 0.836646	valid_1's binary_logloss: 0.139557
    [29]	valid_0's auc: 0.887946	valid_0's binary_logloss: 0.119804	valid_1's auc: 0.836735	valid_1's binary_logloss: 0.139518
    [30]	valid_0's auc: 0.88898	valid_0's binary_logloss: 0.119416	valid_1's auc: 0.836858	valid_1's binary_logloss: 0.139499
    [31]	valid_0's auc: 0.889792	valid_0's binary_logloss: 0.119058	valid_1's auc: 0.836917	valid_1's binary_logloss: 0.139463
    [32]	valid_0's auc: 0.890876	valid_0's binary_logloss: 0.118631	valid_1's auc: 0.836346	valid_1's binary_logloss: 0.139532
    [33]	valid_0's auc: 0.891629	valid_0's binary_logloss: 0.118259	valid_1's auc: 0.836206	valid_1's binary_logloss: 0.139603
    [34]	valid_0's auc: 0.892446	valid_0's binary_logloss: 0.117893	valid_1's auc: 0.836005	valid_1's binary_logloss: 0.139603
    [35]	valid_0's auc: 0.893407	valid_0's binary_logloss: 0.11752	valid_1's auc: 0.8361	valid_1's binary_logloss: 0.139574
    [36]	valid_0's auc: 0.893836	valid_0's binary_logloss: 0.117247	valid_1's auc: 0.836147	valid_1's binary_logloss: 0.139608
    [37]	valid_0's auc: 0.894774	valid_0's binary_logloss: 0.116913	valid_1's auc: 0.836601	valid_1's binary_logloss: 0.139569
    [38]	valid_0's auc: 0.895494	valid_0's binary_logloss: 0.116611	valid_1's auc: 0.836232	valid_1's binary_logloss: 0.139645
    [39]	valid_0's auc: 0.896102	valid_0's binary_logloss: 0.116275	valid_1's auc: 0.836415	valid_1's binary_logloss: 0.139653
    [40]	valid_0's auc: 0.896715	valid_0's binary_logloss: 0.115934	valid_1's auc: 0.836463	valid_1's binary_logloss: 0.139671
    [41]	valid_0's auc: 0.897232	valid_0's binary_logloss: 0.115612	valid_1's auc: 0.836223	valid_1's binary_logloss: 0.139762
    [42]	valid_0's auc: 0.897875	valid_0's binary_logloss: 0.11528	valid_1's auc: 0.836151	valid_1's binary_logloss: 0.139777
    [43]	valid_0's auc: 0.898493	valid_0's binary_logloss: 0.114999	valid_1's auc: 0.836216	valid_1's binary_logloss: 0.139761
    [44]	valid_0's auc: 0.899179	valid_0's binary_logloss: 0.114703	valid_1's auc: 0.836328	valid_1's binary_logloss: 0.139755
    Early stopping, best iteration is:
    [14]	valid_0's auc: 0.870957	valid_0's binary_logloss: 0.128226	valid_1's auc: 0.838226	valid_1's binary_logloss: 0.141392
    [1]	valid_0's auc: 0.834724	valid_0's binary_logloss: 0.15607	valid_1's auc: 0.822983	valid_1's binary_logloss: 0.165104
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.842835	valid_0's binary_logloss: 0.150494	valid_1's auc: 0.830472	valid_1's binary_logloss: 0.159671
    [3]	valid_0's auc: 0.847187	valid_0's binary_logloss: 0.146306	valid_1's auc: 0.830873	valid_1's binary_logloss: 0.155985
    [4]	valid_0's auc: 0.850394	valid_0's binary_logloss: 0.143088	valid_1's auc: 0.830975	valid_1's binary_logloss: 0.15321
    [5]	valid_0's auc: 0.853379	valid_0's binary_logloss: 0.140508	valid_1's auc: 0.832135	valid_1's binary_logloss: 0.150854
    [6]	valid_0's auc: 0.855463	valid_0's binary_logloss: 0.138297	valid_1's auc: 0.833116	valid_1's binary_logloss: 0.149013
    [7]	valid_0's auc: 0.856723	valid_0's binary_logloss: 0.136504	valid_1's auc: 0.833811	valid_1's binary_logloss: 0.147577
    [8]	valid_0's auc: 0.858076	valid_0's binary_logloss: 0.13495	valid_1's auc: 0.835315	valid_1's binary_logloss: 0.146273
    [9]	valid_0's auc: 0.861024	valid_0's binary_logloss: 0.133583	valid_1's auc: 0.835042	valid_1's binary_logloss: 0.145374
    [10]	valid_0's auc: 0.862281	valid_0's binary_logloss: 0.132357	valid_1's auc: 0.834154	valid_1's binary_logloss: 0.144649
    [11]	valid_0's auc: 0.864612	valid_0's binary_logloss: 0.131283	valid_1's auc: 0.834587	valid_1's binary_logloss: 0.143941
    [12]	valid_0's auc: 0.866377	valid_0's binary_logloss: 0.130299	valid_1's auc: 0.834242	valid_1's binary_logloss: 0.143366
    [13]	valid_0's auc: 0.868343	valid_0's binary_logloss: 0.129417	valid_1's auc: 0.833273	valid_1's binary_logloss: 0.142976
    [14]	valid_0's auc: 0.86957	valid_0's binary_logloss: 0.128593	valid_1's auc: 0.833783	valid_1's binary_logloss: 0.142567
    [15]	valid_0's auc: 0.871109	valid_0's binary_logloss: 0.127759	valid_1's auc: 0.834057	valid_1's binary_logloss: 0.142234
    [16]	valid_0's auc: 0.872893	valid_0's binary_logloss: 0.126996	valid_1's auc: 0.835329	valid_1's binary_logloss: 0.141809
    [17]	valid_0's auc: 0.874236	valid_0's binary_logloss: 0.12631	valid_1's auc: 0.834985	valid_1's binary_logloss: 0.141613
    [18]	valid_0's auc: 0.875324	valid_0's binary_logloss: 0.125725	valid_1's auc: 0.834942	valid_1's binary_logloss: 0.141363
    [19]	valid_0's auc: 0.876659	valid_0's binary_logloss: 0.125068	valid_1's auc: 0.835024	valid_1's binary_logloss: 0.141162
    [20]	valid_0's auc: 0.877885	valid_0's binary_logloss: 0.124484	valid_1's auc: 0.835893	valid_1's binary_logloss: 0.140933
    [21]	valid_0's auc: 0.879121	valid_0's binary_logloss: 0.12391	valid_1's auc: 0.837029	valid_1's binary_logloss: 0.140651
    [22]	valid_0's auc: 0.880116	valid_0's binary_logloss: 0.123339	valid_1's auc: 0.837366	valid_1's binary_logloss: 0.140547
    [23]	valid_0's auc: 0.881224	valid_0's binary_logloss: 0.12282	valid_1's auc: 0.837357	valid_1's binary_logloss: 0.140445
    [24]	valid_0's auc: 0.882014	valid_0's binary_logloss: 0.122386	valid_1's auc: 0.837343	valid_1's binary_logloss: 0.140371
    [25]	valid_0's auc: 0.88318	valid_0's binary_logloss: 0.121861	valid_1's auc: 0.83723	valid_1's binary_logloss: 0.140313
    [26]	valid_0's auc: 0.884008	valid_0's binary_logloss: 0.121441	valid_1's auc: 0.837761	valid_1's binary_logloss: 0.140173
    [27]	valid_0's auc: 0.884676	valid_0's binary_logloss: 0.121001	valid_1's auc: 0.838046	valid_1's binary_logloss: 0.140086
    [28]	valid_0's auc: 0.885524	valid_0's binary_logloss: 0.120598	valid_1's auc: 0.838029	valid_1's binary_logloss: 0.140051
    [29]	valid_0's auc: 0.886461	valid_0's binary_logloss: 0.120157	valid_1's auc: 0.837775	valid_1's binary_logloss: 0.140057
    [30]	valid_0's auc: 0.887053	valid_0's binary_logloss: 0.119807	valid_1's auc: 0.837472	valid_1's binary_logloss: 0.140111
    [31]	valid_0's auc: 0.888177	valid_0's binary_logloss: 0.119425	valid_1's auc: 0.837575	valid_1's binary_logloss: 0.140093
    [32]	valid_0's auc: 0.889072	valid_0's binary_logloss: 0.119055	valid_1's auc: 0.837158	valid_1's binary_logloss: 0.140195
    [33]	valid_0's auc: 0.889782	valid_0's binary_logloss: 0.118676	valid_1's auc: 0.837296	valid_1's binary_logloss: 0.140221
    [34]	valid_0's auc: 0.890876	valid_0's binary_logloss: 0.118304	valid_1's auc: 0.837481	valid_1's binary_logloss: 0.140165
    [35]	valid_0's auc: 0.891448	valid_0's binary_logloss: 0.11798	valid_1's auc: 0.837953	valid_1's binary_logloss: 0.140085
    [36]	valid_0's auc: 0.892165	valid_0's binary_logloss: 0.11764	valid_1's auc: 0.837794	valid_1's binary_logloss: 0.140112
    [37]	valid_0's auc: 0.892798	valid_0's binary_logloss: 0.117321	valid_1's auc: 0.837291	valid_1's binary_logloss: 0.140221
    [38]	valid_0's auc: 0.893318	valid_0's binary_logloss: 0.117028	valid_1's auc: 0.837278	valid_1's binary_logloss: 0.140221
    [39]	valid_0's auc: 0.894018	valid_0's binary_logloss: 0.116742	valid_1's auc: 0.83724	valid_1's binary_logloss: 0.140232
    [40]	valid_0's auc: 0.894781	valid_0's binary_logloss: 0.116373	valid_1's auc: 0.836901	valid_1's binary_logloss: 0.140328
    [41]	valid_0's auc: 0.895222	valid_0's binary_logloss: 0.116075	valid_1's auc: 0.836655	valid_1's binary_logloss: 0.140422
    [42]	valid_0's auc: 0.895842	valid_0's binary_logloss: 0.115755	valid_1's auc: 0.836383	valid_1's binary_logloss: 0.140503
    [43]	valid_0's auc: 0.896389	valid_0's binary_logloss: 0.115503	valid_1's auc: 0.836348	valid_1's binary_logloss: 0.140505
    [44]	valid_0's auc: 0.896843	valid_0's binary_logloss: 0.115204	valid_1's auc: 0.836521	valid_1's binary_logloss: 0.140518
    [45]	valid_0's auc: 0.897272	valid_0's binary_logloss: 0.114886	valid_1's auc: 0.836311	valid_1's binary_logloss: 0.140581
    [46]	valid_0's auc: 0.898034	valid_0's binary_logloss: 0.114544	valid_1's auc: 0.835871	valid_1's binary_logloss: 0.140663
    [47]	valid_0's auc: 0.898562	valid_0's binary_logloss: 0.114262	valid_1's auc: 0.835926	valid_1's binary_logloss: 0.140642
    [48]	valid_0's auc: 0.898919	valid_0's binary_logloss: 0.114006	valid_1's auc: 0.835849	valid_1's binary_logloss: 0.140687
    [49]	valid_0's auc: 0.899111	valid_0's binary_logloss: 0.113791	valid_1's auc: 0.835874	valid_1's binary_logloss: 0.140728
    [50]	valid_0's auc: 0.89987	valid_0's binary_logloss: 0.113543	valid_1's auc: 0.835915	valid_1's binary_logloss: 0.14075
    [51]	valid_0's auc: 0.90004	valid_0's binary_logloss: 0.113342	valid_1's auc: 0.835947	valid_1's binary_logloss: 0.140748
    [52]	valid_0's auc: 0.900405	valid_0's binary_logloss: 0.113087	valid_1's auc: 0.836011	valid_1's binary_logloss: 0.140767
    [53]	valid_0's auc: 0.900828	valid_0's binary_logloss: 0.112831	valid_1's auc: 0.836259	valid_1's binary_logloss: 0.140771
    [54]	valid_0's auc: 0.901597	valid_0's binary_logloss: 0.112604	valid_1's auc: 0.836296	valid_1's binary_logloss: 0.14078
    [55]	valid_0's auc: 0.901645	valid_0's binary_logloss: 0.112429	valid_1's auc: 0.836095	valid_1's binary_logloss: 0.140822
    [56]	valid_0's auc: 0.902162	valid_0's binary_logloss: 0.112169	valid_1's auc: 0.835965	valid_1's binary_logloss: 0.14086
    [57]	valid_0's auc: 0.902422	valid_0's binary_logloss: 0.111944	valid_1's auc: 0.835493	valid_1's binary_logloss: 0.140993
    Early stopping, best iteration is:
    [27]	valid_0's auc: 0.884676	valid_0's binary_logloss: 0.121001	valid_1's auc: 0.838046	valid_1's binary_logloss: 0.140086
    [1]	valid_0's auc: 0.820235	valid_0's binary_logloss: 0.156085	valid_1's auc: 0.81613	valid_1's binary_logloss: 0.164992
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.825778	valid_0's binary_logloss: 0.150951	valid_1's auc: 0.821835	valid_1's binary_logloss: 0.159874
    [3]	valid_0's auc: 0.832262	valid_0's binary_logloss: 0.147158	valid_1's auc: 0.826533	valid_1's binary_logloss: 0.156346
    [4]	valid_0's auc: 0.83865	valid_0's binary_logloss: 0.144126	valid_1's auc: 0.833166	valid_1's binary_logloss: 0.1534
    [5]	valid_0's auc: 0.842822	valid_0's binary_logloss: 0.141725	valid_1's auc: 0.836448	valid_1's binary_logloss: 0.151167
    [6]	valid_0's auc: 0.844702	valid_0's binary_logloss: 0.139642	valid_1's auc: 0.837094	valid_1's binary_logloss: 0.149356
    [7]	valid_0's auc: 0.847144	valid_0's binary_logloss: 0.13794	valid_1's auc: 0.837965	valid_1's binary_logloss: 0.147853
    [8]	valid_0's auc: 0.848277	valid_0's binary_logloss: 0.136499	valid_1's auc: 0.837663	valid_1's binary_logloss: 0.146543
    [9]	valid_0's auc: 0.849328	valid_0's binary_logloss: 0.135326	valid_1's auc: 0.837413	valid_1's binary_logloss: 0.145528
    [10]	valid_0's auc: 0.851112	valid_0's binary_logloss: 0.134188	valid_1's auc: 0.836954	valid_1's binary_logloss: 0.14466
    [11]	valid_0's auc: 0.852613	valid_0's binary_logloss: 0.133257	valid_1's auc: 0.837393	valid_1's binary_logloss: 0.143843
    [12]	valid_0's auc: 0.854906	valid_0's binary_logloss: 0.132346	valid_1's auc: 0.837459	valid_1's binary_logloss: 0.143285
    [13]	valid_0's auc: 0.855656	valid_0's binary_logloss: 0.131601	valid_1's auc: 0.837612	valid_1's binary_logloss: 0.142732
    [14]	valid_0's auc: 0.857076	valid_0's binary_logloss: 0.130884	valid_1's auc: 0.837055	valid_1's binary_logloss: 0.142403
    [15]	valid_0's auc: 0.857961	valid_0's binary_logloss: 0.130252	valid_1's auc: 0.837198	valid_1's binary_logloss: 0.142031
    [16]	valid_0's auc: 0.860191	valid_0's binary_logloss: 0.129596	valid_1's auc: 0.836016	valid_1's binary_logloss: 0.141822
    [17]	valid_0's auc: 0.860941	valid_0's binary_logloss: 0.129064	valid_1's auc: 0.836076	valid_1's binary_logloss: 0.141551
    [18]	valid_0's auc: 0.862201	valid_0's binary_logloss: 0.128565	valid_1's auc: 0.835929	valid_1's binary_logloss: 0.141326
    [19]	valid_0's auc: 0.863581	valid_0's binary_logloss: 0.128105	valid_1's auc: 0.835256	valid_1's binary_logloss: 0.141243
    [20]	valid_0's auc: 0.864799	valid_0's binary_logloss: 0.127654	valid_1's auc: 0.83435	valid_1's binary_logloss: 0.141148
    [21]	valid_0's auc: 0.866472	valid_0's binary_logloss: 0.127165	valid_1's auc: 0.834176	valid_1's binary_logloss: 0.141041
    [22]	valid_0's auc: 0.867055	valid_0's binary_logloss: 0.126777	valid_1's auc: 0.834173	valid_1's binary_logloss: 0.140887
    [23]	valid_0's auc: 0.867726	valid_0's binary_logloss: 0.12643	valid_1's auc: 0.833577	valid_1's binary_logloss: 0.140909
    [24]	valid_0's auc: 0.868612	valid_0's binary_logloss: 0.126061	valid_1's auc: 0.833336	valid_1's binary_logloss: 0.140824
    [25]	valid_0's auc: 0.869224	valid_0's binary_logloss: 0.125753	valid_1's auc: 0.833428	valid_1's binary_logloss: 0.140793
    [26]	valid_0's auc: 0.870183	valid_0's binary_logloss: 0.125414	valid_1's auc: 0.83333	valid_1's binary_logloss: 0.140724
    [27]	valid_0's auc: 0.870926	valid_0's binary_logloss: 0.125123	valid_1's auc: 0.832503	valid_1's binary_logloss: 0.140772
    [28]	valid_0's auc: 0.872431	valid_0's binary_logloss: 0.124766	valid_1's auc: 0.832826	valid_1's binary_logloss: 0.140685
    [29]	valid_0's auc: 0.873397	valid_0's binary_logloss: 0.124495	valid_1's auc: 0.833175	valid_1's binary_logloss: 0.140604
    [30]	valid_0's auc: 0.87475	valid_0's binary_logloss: 0.12417	valid_1's auc: 0.833614	valid_1's binary_logloss: 0.140497
    [31]	valid_0's auc: 0.875407	valid_0's binary_logloss: 0.12389	valid_1's auc: 0.833706	valid_1's binary_logloss: 0.140428
    [32]	valid_0's auc: 0.876136	valid_0's binary_logloss: 0.123637	valid_1's auc: 0.833458	valid_1's binary_logloss: 0.140448
    [33]	valid_0's auc: 0.876575	valid_0's binary_logloss: 0.123421	valid_1's auc: 0.832965	valid_1's binary_logloss: 0.140498
    [34]	valid_0's auc: 0.877224	valid_0's binary_logloss: 0.123219	valid_1's auc: 0.832659	valid_1's binary_logloss: 0.140537
    [35]	valid_0's auc: 0.877898	valid_0's binary_logloss: 0.122947	valid_1's auc: 0.832787	valid_1's binary_logloss: 0.140536
    [36]	valid_0's auc: 0.878334	valid_0's binary_logloss: 0.122724	valid_1's auc: 0.832724	valid_1's binary_logloss: 0.14053
    [37]	valid_0's auc: 0.878762	valid_0's binary_logloss: 0.122514	valid_1's auc: 0.832581	valid_1's binary_logloss: 0.140533
    Early stopping, best iteration is:
    [7]	valid_0's auc: 0.847144	valid_0's binary_logloss: 0.13794	valid_1's auc: 0.837965	valid_1's binary_logloss: 0.147853
    [1]	valid_0's auc: 0.814371	valid_0's binary_logloss: 0.156452	valid_1's auc: 0.813175	valid_1's binary_logloss: 0.165418
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.827277	valid_0's binary_logloss: 0.151084	valid_1's auc: 0.819635	valid_1's binary_logloss: 0.160159
    [3]	valid_0's auc: 0.837033	valid_0's binary_logloss: 0.14722	valid_1's auc: 0.828221	valid_1's binary_logloss: 0.156492
    [4]	valid_0's auc: 0.840167	valid_0's binary_logloss: 0.14423	valid_1's auc: 0.830942	valid_1's binary_logloss: 0.153586
    [5]	valid_0's auc: 0.842499	valid_0's binary_logloss: 0.141721	valid_1's auc: 0.833301	valid_1's binary_logloss: 0.151219
    [6]	valid_0's auc: 0.845403	valid_0's binary_logloss: 0.139708	valid_1's auc: 0.836412	valid_1's binary_logloss: 0.149312
    [7]	valid_0's auc: 0.848049	valid_0's binary_logloss: 0.138024	valid_1's auc: 0.836054	valid_1's binary_logloss: 0.14779
    [8]	valid_0's auc: 0.849694	valid_0's binary_logloss: 0.136542	valid_1's auc: 0.837537	valid_1's binary_logloss: 0.146417
    [9]	valid_0's auc: 0.851646	valid_0's binary_logloss: 0.135289	valid_1's auc: 0.838418	valid_1's binary_logloss: 0.145329
    [10]	valid_0's auc: 0.853642	valid_0's binary_logloss: 0.134189	valid_1's auc: 0.839342	valid_1's binary_logloss: 0.144374
    [11]	valid_0's auc: 0.855647	valid_0's binary_logloss: 0.133227	valid_1's auc: 0.840035	valid_1's binary_logloss: 0.143552
    [12]	valid_0's auc: 0.856768	valid_0's binary_logloss: 0.132399	valid_1's auc: 0.839294	valid_1's binary_logloss: 0.143047
    [13]	valid_0's auc: 0.85763	valid_0's binary_logloss: 0.13165	valid_1's auc: 0.838911	valid_1's binary_logloss: 0.142469
    [14]	valid_0's auc: 0.859243	valid_0's binary_logloss: 0.130936	valid_1's auc: 0.838705	valid_1's binary_logloss: 0.141913
    [15]	valid_0's auc: 0.860124	valid_0's binary_logloss: 0.130312	valid_1's auc: 0.838608	valid_1's binary_logloss: 0.141547
    [16]	valid_0's auc: 0.861358	valid_0's binary_logloss: 0.129687	valid_1's auc: 0.838422	valid_1's binary_logloss: 0.141134
    [17]	valid_0's auc: 0.862159	valid_0's binary_logloss: 0.129139	valid_1's auc: 0.838636	valid_1's binary_logloss: 0.140786
    [18]	valid_0's auc: 0.862729	valid_0's binary_logloss: 0.128664	valid_1's auc: 0.838342	valid_1's binary_logloss: 0.140538
    [19]	valid_0's auc: 0.863842	valid_0's binary_logloss: 0.128137	valid_1's auc: 0.838464	valid_1's binary_logloss: 0.140331
    [20]	valid_0's auc: 0.864859	valid_0's binary_logloss: 0.127657	valid_1's auc: 0.837832	valid_1's binary_logloss: 0.140179
    [21]	valid_0's auc: 0.866227	valid_0's binary_logloss: 0.127137	valid_1's auc: 0.837735	valid_1's binary_logloss: 0.140043
    [22]	valid_0's auc: 0.866925	valid_0's binary_logloss: 0.126772	valid_1's auc: 0.838268	valid_1's binary_logloss: 0.139927
    [23]	valid_0's auc: 0.867727	valid_0's binary_logloss: 0.126369	valid_1's auc: 0.838482	valid_1's binary_logloss: 0.139787
    [24]	valid_0's auc: 0.868239	valid_0's binary_logloss: 0.126013	valid_1's auc: 0.838767	valid_1's binary_logloss: 0.13964
    [25]	valid_0's auc: 0.869304	valid_0's binary_logloss: 0.125622	valid_1's auc: 0.838562	valid_1's binary_logloss: 0.139648
    [26]	valid_0's auc: 0.870347	valid_0's binary_logloss: 0.125288	valid_1's auc: 0.838228	valid_1's binary_logloss: 0.139618
    [27]	valid_0's auc: 0.871198	valid_0's binary_logloss: 0.124953	valid_1's auc: 0.838403	valid_1's binary_logloss: 0.139594
    [28]	valid_0's auc: 0.872024	valid_0's binary_logloss: 0.124672	valid_1's auc: 0.838405	valid_1's binary_logloss: 0.139526
    [29]	valid_0's auc: 0.873184	valid_0's binary_logloss: 0.124303	valid_1's auc: 0.838211	valid_1's binary_logloss: 0.139531
    [30]	valid_0's auc: 0.874076	valid_0's binary_logloss: 0.12403	valid_1's auc: 0.838983	valid_1's binary_logloss: 0.139411
    [31]	valid_0's auc: 0.874768	valid_0's binary_logloss: 0.123745	valid_1's auc: 0.839314	valid_1's binary_logloss: 0.139314
    [32]	valid_0's auc: 0.875593	valid_0's binary_logloss: 0.123486	valid_1's auc: 0.838875	valid_1's binary_logloss: 0.139322
    [33]	valid_0's auc: 0.8767	valid_0's binary_logloss: 0.123182	valid_1's auc: 0.838809	valid_1's binary_logloss: 0.139329
    [34]	valid_0's auc: 0.87774	valid_0's binary_logloss: 0.122892	valid_1's auc: 0.838376	valid_1's binary_logloss: 0.139342
    [35]	valid_0's auc: 0.878372	valid_0's binary_logloss: 0.122634	valid_1's auc: 0.838454	valid_1's binary_logloss: 0.13931
    [36]	valid_0's auc: 0.879098	valid_0's binary_logloss: 0.122414	valid_1's auc: 0.838895	valid_1's binary_logloss: 0.13925
    [37]	valid_0's auc: 0.879502	valid_0's binary_logloss: 0.122216	valid_1's auc: 0.838441	valid_1's binary_logloss: 0.139302
    [38]	valid_0's auc: 0.880036	valid_0's binary_logloss: 0.121998	valid_1's auc: 0.838582	valid_1's binary_logloss: 0.139306
    [39]	valid_0's auc: 0.880641	valid_0's binary_logloss: 0.121716	valid_1's auc: 0.838787	valid_1's binary_logloss: 0.139269
    [40]	valid_0's auc: 0.881249	valid_0's binary_logloss: 0.121482	valid_1's auc: 0.838906	valid_1's binary_logloss: 0.139223
    [41]	valid_0's auc: 0.881919	valid_0's binary_logloss: 0.121223	valid_1's auc: 0.838567	valid_1's binary_logloss: 0.13926
    Early stopping, best iteration is:
    [11]	valid_0's auc: 0.855647	valid_0's binary_logloss: 0.133227	valid_1's auc: 0.840035	valid_1's binary_logloss: 0.143552
    [1]	valid_0's auc: 0.821645	valid_0's binary_logloss: 0.156528	valid_1's auc: 0.81857	valid_1's binary_logloss: 0.165101
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.827488	valid_0's binary_logloss: 0.151189	valid_1's auc: 0.822299	valid_1's binary_logloss: 0.160072
    [3]	valid_0's auc: 0.837855	valid_0's binary_logloss: 0.147263	valid_1's auc: 0.829855	valid_1's binary_logloss: 0.156527
    [4]	valid_0's auc: 0.840063	valid_0's binary_logloss: 0.144261	valid_1's auc: 0.833088	valid_1's binary_logloss: 0.153446
    [5]	valid_0's auc: 0.842802	valid_0's binary_logloss: 0.141691	valid_1's auc: 0.834541	valid_1's binary_logloss: 0.151144
    [6]	valid_0's auc: 0.844	valid_0's binary_logloss: 0.139654	valid_1's auc: 0.834542	valid_1's binary_logloss: 0.149333
    [7]	valid_0's auc: 0.845838	valid_0's binary_logloss: 0.138002	valid_1's auc: 0.835645	valid_1's binary_logloss: 0.147676
    [8]	valid_0's auc: 0.846869	valid_0's binary_logloss: 0.136628	valid_1's auc: 0.836118	valid_1's binary_logloss: 0.146491
    [9]	valid_0's auc: 0.849282	valid_0's binary_logloss: 0.135382	valid_1's auc: 0.837542	valid_1's binary_logloss: 0.14539
    [10]	valid_0's auc: 0.851021	valid_0's binary_logloss: 0.134282	valid_1's auc: 0.836942	valid_1's binary_logloss: 0.144584
    [11]	valid_0's auc: 0.852037	valid_0's binary_logloss: 0.133358	valid_1's auc: 0.8374	valid_1's binary_logloss: 0.143836
    [12]	valid_0's auc: 0.854496	valid_0's binary_logloss: 0.132505	valid_1's auc: 0.838593	valid_1's binary_logloss: 0.143171
    [13]	valid_0's auc: 0.857514	valid_0's binary_logloss: 0.131695	valid_1's auc: 0.838558	valid_1's binary_logloss: 0.142646
    [14]	valid_0's auc: 0.858827	valid_0's binary_logloss: 0.131006	valid_1's auc: 0.838498	valid_1's binary_logloss: 0.142158
    [15]	valid_0's auc: 0.860574	valid_0's binary_logloss: 0.130352	valid_1's auc: 0.837435	valid_1's binary_logloss: 0.141868
    [16]	valid_0's auc: 0.861239	valid_0's binary_logloss: 0.129765	valid_1's auc: 0.837374	valid_1's binary_logloss: 0.141537
    [17]	valid_0's auc: 0.86217	valid_0's binary_logloss: 0.129164	valid_1's auc: 0.837703	valid_1's binary_logloss: 0.141192
    [18]	valid_0's auc: 0.863228	valid_0's binary_logloss: 0.128615	valid_1's auc: 0.837526	valid_1's binary_logloss: 0.140917
    [19]	valid_0's auc: 0.86473	valid_0's binary_logloss: 0.128113	valid_1's auc: 0.838235	valid_1's binary_logloss: 0.140572
    [20]	valid_0's auc: 0.865797	valid_0's binary_logloss: 0.127679	valid_1's auc: 0.838788	valid_1's binary_logloss: 0.140332
    [21]	valid_0's auc: 0.866561	valid_0's binary_logloss: 0.127235	valid_1's auc: 0.839171	valid_1's binary_logloss: 0.140108
    [22]	valid_0's auc: 0.867237	valid_0's binary_logloss: 0.12688	valid_1's auc: 0.839213	valid_1's binary_logloss: 0.13991
    [23]	valid_0's auc: 0.867894	valid_0's binary_logloss: 0.126519	valid_1's auc: 0.839641	valid_1's binary_logloss: 0.139745
    [24]	valid_0's auc: 0.868501	valid_0's binary_logloss: 0.126192	valid_1's auc: 0.840025	valid_1's binary_logloss: 0.139593
    [25]	valid_0's auc: 0.869311	valid_0's binary_logloss: 0.125838	valid_1's auc: 0.839961	valid_1's binary_logloss: 0.139531
    [26]	valid_0's auc: 0.870325	valid_0's binary_logloss: 0.125518	valid_1's auc: 0.839261	valid_1's binary_logloss: 0.139524
    [27]	valid_0's auc: 0.871488	valid_0's binary_logloss: 0.125147	valid_1's auc: 0.839671	valid_1's binary_logloss: 0.139365
    [28]	valid_0's auc: 0.87235	valid_0's binary_logloss: 0.12484	valid_1's auc: 0.840114	valid_1's binary_logloss: 0.139236
    [29]	valid_0's auc: 0.872991	valid_0's binary_logloss: 0.124593	valid_1's auc: 0.839491	valid_1's binary_logloss: 0.139271
    [30]	valid_0's auc: 0.874129	valid_0's binary_logloss: 0.124312	valid_1's auc: 0.839589	valid_1's binary_logloss: 0.13918
    [31]	valid_0's auc: 0.875305	valid_0's binary_logloss: 0.123988	valid_1's auc: 0.839441	valid_1's binary_logloss: 0.139184
    [32]	valid_0's auc: 0.875943	valid_0's binary_logloss: 0.123748	valid_1's auc: 0.839268	valid_1's binary_logloss: 0.13919
    [33]	valid_0's auc: 0.876575	valid_0's binary_logloss: 0.123484	valid_1's auc: 0.839549	valid_1's binary_logloss: 0.139075
    [34]	valid_0's auc: 0.877426	valid_0's binary_logloss: 0.123156	valid_1's auc: 0.839087	valid_1's binary_logloss: 0.139148
    [35]	valid_0's auc: 0.87822	valid_0's binary_logloss: 0.122873	valid_1's auc: 0.8389	valid_1's binary_logloss: 0.139187
    [36]	valid_0's auc: 0.878932	valid_0's binary_logloss: 0.12259	valid_1's auc: 0.838921	valid_1's binary_logloss: 0.139194
    [37]	valid_0's auc: 0.879842	valid_0's binary_logloss: 0.12233	valid_1's auc: 0.839133	valid_1's binary_logloss: 0.139161
    [38]	valid_0's auc: 0.880497	valid_0's binary_logloss: 0.12208	valid_1's auc: 0.838975	valid_1's binary_logloss: 0.139143
    [39]	valid_0's auc: 0.881056	valid_0's binary_logloss: 0.121827	valid_1's auc: 0.839037	valid_1's binary_logloss: 0.139138
    [40]	valid_0's auc: 0.881604	valid_0's binary_logloss: 0.121603	valid_1's auc: 0.839204	valid_1's binary_logloss: 0.139119
    [41]	valid_0's auc: 0.882159	valid_0's binary_logloss: 0.121355	valid_1's auc: 0.839277	valid_1's binary_logloss: 0.139091
    [42]	valid_0's auc: 0.882757	valid_0's binary_logloss: 0.121116	valid_1's auc: 0.838964	valid_1's binary_logloss: 0.139133
    [43]	valid_0's auc: 0.883143	valid_0's binary_logloss: 0.120918	valid_1's auc: 0.839024	valid_1's binary_logloss: 0.139124
    [44]	valid_0's auc: 0.883697	valid_0's binary_logloss: 0.12072	valid_1's auc: 0.838652	valid_1's binary_logloss: 0.139203
    [45]	valid_0's auc: 0.884292	valid_0's binary_logloss: 0.120482	valid_1's auc: 0.839016	valid_1's binary_logloss: 0.139124
    [46]	valid_0's auc: 0.884969	valid_0's binary_logloss: 0.120266	valid_1's auc: 0.838683	valid_1's binary_logloss: 0.139184
    [47]	valid_0's auc: 0.8853	valid_0's binary_logloss: 0.120089	valid_1's auc: 0.838624	valid_1's binary_logloss: 0.139193
    [48]	valid_0's auc: 0.885876	valid_0's binary_logloss: 0.11993	valid_1's auc: 0.838569	valid_1's binary_logloss: 0.139212
    [49]	valid_0's auc: 0.886141	valid_0's binary_logloss: 0.119757	valid_1's auc: 0.838345	valid_1's binary_logloss: 0.139288
    [50]	valid_0's auc: 0.886433	valid_0's binary_logloss: 0.119595	valid_1's auc: 0.838342	valid_1's binary_logloss: 0.139332
    [51]	valid_0's auc: 0.886975	valid_0's binary_logloss: 0.119377	valid_1's auc: 0.838335	valid_1's binary_logloss: 0.139331
    [52]	valid_0's auc: 0.887568	valid_0's binary_logloss: 0.119161	valid_1's auc: 0.838204	valid_1's binary_logloss: 0.139331
    [53]	valid_0's auc: 0.887867	valid_0's binary_logloss: 0.118974	valid_1's auc: 0.838044	valid_1's binary_logloss: 0.13936
    [54]	valid_0's auc: 0.888093	valid_0's binary_logloss: 0.118834	valid_1's auc: 0.838137	valid_1's binary_logloss: 0.13935
    [55]	valid_0's auc: 0.888289	valid_0's binary_logloss: 0.118675	valid_1's auc: 0.837878	valid_1's binary_logloss: 0.139392
    [56]	valid_0's auc: 0.888615	valid_0's binary_logloss: 0.118561	valid_1's auc: 0.837776	valid_1's binary_logloss: 0.139418
    [57]	valid_0's auc: 0.889157	valid_0's binary_logloss: 0.118369	valid_1's auc: 0.837585	valid_1's binary_logloss: 0.139447
    [58]	valid_0's auc: 0.889659	valid_0's binary_logloss: 0.11819	valid_1's auc: 0.837789	valid_1's binary_logloss: 0.139431
    Early stopping, best iteration is:
    [28]	valid_0's auc: 0.87235	valid_0's binary_logloss: 0.12484	valid_1's auc: 0.840114	valid_1's binary_logloss: 0.139236
    [1]	valid_0's auc: 0.820235	valid_0's binary_logloss: 0.156085	valid_1's auc: 0.81613	valid_1's binary_logloss: 0.164992
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.825778	valid_0's binary_logloss: 0.150951	valid_1's auc: 0.821835	valid_1's binary_logloss: 0.159874
    [3]	valid_0's auc: 0.832262	valid_0's binary_logloss: 0.147158	valid_1's auc: 0.826533	valid_1's binary_logloss: 0.156346
    [4]	valid_0's auc: 0.83865	valid_0's binary_logloss: 0.144126	valid_1's auc: 0.833166	valid_1's binary_logloss: 0.1534
    [5]	valid_0's auc: 0.842822	valid_0's binary_logloss: 0.141725	valid_1's auc: 0.836448	valid_1's binary_logloss: 0.151167
    [6]	valid_0's auc: 0.844702	valid_0's binary_logloss: 0.139642	valid_1's auc: 0.837094	valid_1's binary_logloss: 0.149356
    [7]	valid_0's auc: 0.847144	valid_0's binary_logloss: 0.13794	valid_1's auc: 0.837965	valid_1's binary_logloss: 0.147853
    [8]	valid_0's auc: 0.848277	valid_0's binary_logloss: 0.136499	valid_1's auc: 0.837663	valid_1's binary_logloss: 0.146543
    [9]	valid_0's auc: 0.849328	valid_0's binary_logloss: 0.135326	valid_1's auc: 0.837413	valid_1's binary_logloss: 0.145528
    [10]	valid_0's auc: 0.851112	valid_0's binary_logloss: 0.134188	valid_1's auc: 0.836954	valid_1's binary_logloss: 0.14466
    [11]	valid_0's auc: 0.852613	valid_0's binary_logloss: 0.133257	valid_1's auc: 0.837393	valid_1's binary_logloss: 0.143843
    [12]	valid_0's auc: 0.854906	valid_0's binary_logloss: 0.132346	valid_1's auc: 0.837459	valid_1's binary_logloss: 0.143285
    [13]	valid_0's auc: 0.855656	valid_0's binary_logloss: 0.131601	valid_1's auc: 0.837612	valid_1's binary_logloss: 0.142732
    [14]	valid_0's auc: 0.857076	valid_0's binary_logloss: 0.130884	valid_1's auc: 0.837055	valid_1's binary_logloss: 0.142403
    [15]	valid_0's auc: 0.857961	valid_0's binary_logloss: 0.130252	valid_1's auc: 0.837198	valid_1's binary_logloss: 0.142031
    [16]	valid_0's auc: 0.860191	valid_0's binary_logloss: 0.129596	valid_1's auc: 0.836016	valid_1's binary_logloss: 0.141822
    [17]	valid_0's auc: 0.860941	valid_0's binary_logloss: 0.129064	valid_1's auc: 0.836076	valid_1's binary_logloss: 0.141551
    [18]	valid_0's auc: 0.862201	valid_0's binary_logloss: 0.128565	valid_1's auc: 0.835929	valid_1's binary_logloss: 0.141326
    [19]	valid_0's auc: 0.863581	valid_0's binary_logloss: 0.128105	valid_1's auc: 0.835256	valid_1's binary_logloss: 0.141243
    [20]	valid_0's auc: 0.864799	valid_0's binary_logloss: 0.127654	valid_1's auc: 0.83435	valid_1's binary_logloss: 0.141148
    [21]	valid_0's auc: 0.866472	valid_0's binary_logloss: 0.127165	valid_1's auc: 0.834176	valid_1's binary_logloss: 0.141041
    [22]	valid_0's auc: 0.867055	valid_0's binary_logloss: 0.126777	valid_1's auc: 0.834173	valid_1's binary_logloss: 0.140887
    [23]	valid_0's auc: 0.867726	valid_0's binary_logloss: 0.12643	valid_1's auc: 0.833577	valid_1's binary_logloss: 0.140909
    [24]	valid_0's auc: 0.868612	valid_0's binary_logloss: 0.126061	valid_1's auc: 0.833336	valid_1's binary_logloss: 0.140824
    [25]	valid_0's auc: 0.869224	valid_0's binary_logloss: 0.125753	valid_1's auc: 0.833428	valid_1's binary_logloss: 0.140793
    [26]	valid_0's auc: 0.870183	valid_0's binary_logloss: 0.125414	valid_1's auc: 0.83333	valid_1's binary_logloss: 0.140724
    [27]	valid_0's auc: 0.870926	valid_0's binary_logloss: 0.125123	valid_1's auc: 0.832503	valid_1's binary_logloss: 0.140772
    [28]	valid_0's auc: 0.872431	valid_0's binary_logloss: 0.124766	valid_1's auc: 0.832826	valid_1's binary_logloss: 0.140685
    [29]	valid_0's auc: 0.873397	valid_0's binary_logloss: 0.124495	valid_1's auc: 0.833175	valid_1's binary_logloss: 0.140604
    [30]	valid_0's auc: 0.87475	valid_0's binary_logloss: 0.12417	valid_1's auc: 0.833614	valid_1's binary_logloss: 0.140497
    [31]	valid_0's auc: 0.875407	valid_0's binary_logloss: 0.12389	valid_1's auc: 0.833706	valid_1's binary_logloss: 0.140428
    [32]	valid_0's auc: 0.876136	valid_0's binary_logloss: 0.123637	valid_1's auc: 0.833458	valid_1's binary_logloss: 0.140448
    [33]	valid_0's auc: 0.876575	valid_0's binary_logloss: 0.123421	valid_1's auc: 0.832965	valid_1's binary_logloss: 0.140498
    [34]	valid_0's auc: 0.877224	valid_0's binary_logloss: 0.123219	valid_1's auc: 0.832659	valid_1's binary_logloss: 0.140537
    [35]	valid_0's auc: 0.877898	valid_0's binary_logloss: 0.122947	valid_1's auc: 0.832787	valid_1's binary_logloss: 0.140536
    [36]	valid_0's auc: 0.878334	valid_0's binary_logloss: 0.122724	valid_1's auc: 0.832724	valid_1's binary_logloss: 0.14053
    [37]	valid_0's auc: 0.878762	valid_0's binary_logloss: 0.122514	valid_1's auc: 0.832581	valid_1's binary_logloss: 0.140533
    Early stopping, best iteration is:
    [7]	valid_0's auc: 0.847144	valid_0's binary_logloss: 0.13794	valid_1's auc: 0.837965	valid_1's binary_logloss: 0.147853
    [1]	valid_0's auc: 0.814371	valid_0's binary_logloss: 0.156452	valid_1's auc: 0.813175	valid_1's binary_logloss: 0.165418
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.827277	valid_0's binary_logloss: 0.151084	valid_1's auc: 0.819635	valid_1's binary_logloss: 0.160159
    [3]	valid_0's auc: 0.837033	valid_0's binary_logloss: 0.14722	valid_1's auc: 0.828221	valid_1's binary_logloss: 0.156492
    [4]	valid_0's auc: 0.840167	valid_0's binary_logloss: 0.14423	valid_1's auc: 0.830942	valid_1's binary_logloss: 0.153586
    [5]	valid_0's auc: 0.842499	valid_0's binary_logloss: 0.141721	valid_1's auc: 0.833301	valid_1's binary_logloss: 0.151219
    [6]	valid_0's auc: 0.845403	valid_0's binary_logloss: 0.139708	valid_1's auc: 0.836412	valid_1's binary_logloss: 0.149312
    [7]	valid_0's auc: 0.848049	valid_0's binary_logloss: 0.138024	valid_1's auc: 0.836054	valid_1's binary_logloss: 0.14779
    [8]	valid_0's auc: 0.849694	valid_0's binary_logloss: 0.136542	valid_1's auc: 0.837537	valid_1's binary_logloss: 0.146417
    [9]	valid_0's auc: 0.851646	valid_0's binary_logloss: 0.135289	valid_1's auc: 0.838418	valid_1's binary_logloss: 0.145329
    [10]	valid_0's auc: 0.853642	valid_0's binary_logloss: 0.134189	valid_1's auc: 0.839342	valid_1's binary_logloss: 0.144374
    [11]	valid_0's auc: 0.855647	valid_0's binary_logloss: 0.133227	valid_1's auc: 0.840035	valid_1's binary_logloss: 0.143552
    [12]	valid_0's auc: 0.856768	valid_0's binary_logloss: 0.132399	valid_1's auc: 0.839294	valid_1's binary_logloss: 0.143047
    [13]	valid_0's auc: 0.85763	valid_0's binary_logloss: 0.13165	valid_1's auc: 0.838911	valid_1's binary_logloss: 0.142469
    [14]	valid_0's auc: 0.859243	valid_0's binary_logloss: 0.130936	valid_1's auc: 0.838705	valid_1's binary_logloss: 0.141913
    [15]	valid_0's auc: 0.860124	valid_0's binary_logloss: 0.130312	valid_1's auc: 0.838608	valid_1's binary_logloss: 0.141547
    [16]	valid_0's auc: 0.861358	valid_0's binary_logloss: 0.129687	valid_1's auc: 0.838422	valid_1's binary_logloss: 0.141134
    [17]	valid_0's auc: 0.862159	valid_0's binary_logloss: 0.129139	valid_1's auc: 0.838636	valid_1's binary_logloss: 0.140786
    [18]	valid_0's auc: 0.862729	valid_0's binary_logloss: 0.128664	valid_1's auc: 0.838342	valid_1's binary_logloss: 0.140538
    [19]	valid_0's auc: 0.863842	valid_0's binary_logloss: 0.128137	valid_1's auc: 0.838464	valid_1's binary_logloss: 0.140331
    [20]	valid_0's auc: 0.864859	valid_0's binary_logloss: 0.127657	valid_1's auc: 0.837832	valid_1's binary_logloss: 0.140179
    [21]	valid_0's auc: 0.866227	valid_0's binary_logloss: 0.127137	valid_1's auc: 0.837735	valid_1's binary_logloss: 0.140043
    [22]	valid_0's auc: 0.866925	valid_0's binary_logloss: 0.126772	valid_1's auc: 0.838268	valid_1's binary_logloss: 0.139927
    [23]	valid_0's auc: 0.867727	valid_0's binary_logloss: 0.126369	valid_1's auc: 0.838482	valid_1's binary_logloss: 0.139787
    [24]	valid_0's auc: 0.868239	valid_0's binary_logloss: 0.126013	valid_1's auc: 0.838767	valid_1's binary_logloss: 0.13964
    [25]	valid_0's auc: 0.869304	valid_0's binary_logloss: 0.125622	valid_1's auc: 0.838562	valid_1's binary_logloss: 0.139648
    [26]	valid_0's auc: 0.870347	valid_0's binary_logloss: 0.125288	valid_1's auc: 0.838228	valid_1's binary_logloss: 0.139618
    [27]	valid_0's auc: 0.871198	valid_0's binary_logloss: 0.124953	valid_1's auc: 0.838403	valid_1's binary_logloss: 0.139594
    [28]	valid_0's auc: 0.872024	valid_0's binary_logloss: 0.124672	valid_1's auc: 0.838405	valid_1's binary_logloss: 0.139526
    [29]	valid_0's auc: 0.873184	valid_0's binary_logloss: 0.124303	valid_1's auc: 0.838211	valid_1's binary_logloss: 0.139531
    [30]	valid_0's auc: 0.874076	valid_0's binary_logloss: 0.12403	valid_1's auc: 0.838983	valid_1's binary_logloss: 0.139411
    [31]	valid_0's auc: 0.874768	valid_0's binary_logloss: 0.123745	valid_1's auc: 0.839314	valid_1's binary_logloss: 0.139314
    [32]	valid_0's auc: 0.875593	valid_0's binary_logloss: 0.123486	valid_1's auc: 0.838875	valid_1's binary_logloss: 0.139322
    [33]	valid_0's auc: 0.8767	valid_0's binary_logloss: 0.123182	valid_1's auc: 0.838809	valid_1's binary_logloss: 0.139329
    [34]	valid_0's auc: 0.87774	valid_0's binary_logloss: 0.122892	valid_1's auc: 0.838376	valid_1's binary_logloss: 0.139342
    [35]	valid_0's auc: 0.878372	valid_0's binary_logloss: 0.122634	valid_1's auc: 0.838454	valid_1's binary_logloss: 0.13931
    [36]	valid_0's auc: 0.879098	valid_0's binary_logloss: 0.122414	valid_1's auc: 0.838895	valid_1's binary_logloss: 0.13925
    [37]	valid_0's auc: 0.879502	valid_0's binary_logloss: 0.122216	valid_1's auc: 0.838441	valid_1's binary_logloss: 0.139302
    [38]	valid_0's auc: 0.880036	valid_0's binary_logloss: 0.121998	valid_1's auc: 0.838582	valid_1's binary_logloss: 0.139306
    [39]	valid_0's auc: 0.880641	valid_0's binary_logloss: 0.121716	valid_1's auc: 0.838787	valid_1's binary_logloss: 0.139269
    [40]	valid_0's auc: 0.881249	valid_0's binary_logloss: 0.121482	valid_1's auc: 0.838906	valid_1's binary_logloss: 0.139223
    [41]	valid_0's auc: 0.881919	valid_0's binary_logloss: 0.121223	valid_1's auc: 0.838567	valid_1's binary_logloss: 0.13926
    Early stopping, best iteration is:
    [11]	valid_0's auc: 0.855647	valid_0's binary_logloss: 0.133227	valid_1's auc: 0.840035	valid_1's binary_logloss: 0.143552
    [1]	valid_0's auc: 0.821645	valid_0's binary_logloss: 0.156528	valid_1's auc: 0.81857	valid_1's binary_logloss: 0.165101
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.827488	valid_0's binary_logloss: 0.151189	valid_1's auc: 0.822299	valid_1's binary_logloss: 0.160072
    [3]	valid_0's auc: 0.837855	valid_0's binary_logloss: 0.147263	valid_1's auc: 0.829855	valid_1's binary_logloss: 0.156527
    [4]	valid_0's auc: 0.840063	valid_0's binary_logloss: 0.144261	valid_1's auc: 0.833088	valid_1's binary_logloss: 0.153446
    [5]	valid_0's auc: 0.842802	valid_0's binary_logloss: 0.141691	valid_1's auc: 0.834541	valid_1's binary_logloss: 0.151144
    [6]	valid_0's auc: 0.844	valid_0's binary_logloss: 0.139654	valid_1's auc: 0.834542	valid_1's binary_logloss: 0.149333
    [7]	valid_0's auc: 0.845838	valid_0's binary_logloss: 0.138002	valid_1's auc: 0.835645	valid_1's binary_logloss: 0.147676
    [8]	valid_0's auc: 0.846869	valid_0's binary_logloss: 0.136628	valid_1's auc: 0.836118	valid_1's binary_logloss: 0.146491
    [9]	valid_0's auc: 0.849282	valid_0's binary_logloss: 0.135382	valid_1's auc: 0.837542	valid_1's binary_logloss: 0.14539
    [10]	valid_0's auc: 0.851021	valid_0's binary_logloss: 0.134282	valid_1's auc: 0.836942	valid_1's binary_logloss: 0.144584
    [11]	valid_0's auc: 0.852037	valid_0's binary_logloss: 0.133358	valid_1's auc: 0.8374	valid_1's binary_logloss: 0.143836
    [12]	valid_0's auc: 0.854496	valid_0's binary_logloss: 0.132505	valid_1's auc: 0.838593	valid_1's binary_logloss: 0.143171
    [13]	valid_0's auc: 0.857514	valid_0's binary_logloss: 0.131695	valid_1's auc: 0.838558	valid_1's binary_logloss: 0.142646
    [14]	valid_0's auc: 0.858827	valid_0's binary_logloss: 0.131006	valid_1's auc: 0.838498	valid_1's binary_logloss: 0.142158
    [15]	valid_0's auc: 0.860574	valid_0's binary_logloss: 0.130352	valid_1's auc: 0.837435	valid_1's binary_logloss: 0.141868
    [16]	valid_0's auc: 0.861239	valid_0's binary_logloss: 0.129765	valid_1's auc: 0.837374	valid_1's binary_logloss: 0.141537
    [17]	valid_0's auc: 0.86217	valid_0's binary_logloss: 0.129164	valid_1's auc: 0.837703	valid_1's binary_logloss: 0.141192
    [18]	valid_0's auc: 0.863228	valid_0's binary_logloss: 0.128615	valid_1's auc: 0.837526	valid_1's binary_logloss: 0.140917
    [19]	valid_0's auc: 0.86473	valid_0's binary_logloss: 0.128113	valid_1's auc: 0.838235	valid_1's binary_logloss: 0.140572
    [20]	valid_0's auc: 0.865797	valid_0's binary_logloss: 0.127679	valid_1's auc: 0.838788	valid_1's binary_logloss: 0.140332
    [21]	valid_0's auc: 0.866561	valid_0's binary_logloss: 0.127235	valid_1's auc: 0.839171	valid_1's binary_logloss: 0.140108
    [22]	valid_0's auc: 0.867237	valid_0's binary_logloss: 0.12688	valid_1's auc: 0.839213	valid_1's binary_logloss: 0.13991
    [23]	valid_0's auc: 0.867894	valid_0's binary_logloss: 0.126519	valid_1's auc: 0.839641	valid_1's binary_logloss: 0.139745
    [24]	valid_0's auc: 0.868501	valid_0's binary_logloss: 0.126192	valid_1's auc: 0.840025	valid_1's binary_logloss: 0.139593
    [25]	valid_0's auc: 0.869311	valid_0's binary_logloss: 0.125838	valid_1's auc: 0.839961	valid_1's binary_logloss: 0.139531
    [26]	valid_0's auc: 0.870325	valid_0's binary_logloss: 0.125518	valid_1's auc: 0.839261	valid_1's binary_logloss: 0.139524
    [27]	valid_0's auc: 0.871488	valid_0's binary_logloss: 0.125147	valid_1's auc: 0.839671	valid_1's binary_logloss: 0.139365
    [28]	valid_0's auc: 0.87235	valid_0's binary_logloss: 0.12484	valid_1's auc: 0.840114	valid_1's binary_logloss: 0.139236
    [29]	valid_0's auc: 0.872991	valid_0's binary_logloss: 0.124593	valid_1's auc: 0.839491	valid_1's binary_logloss: 0.139271
    [30]	valid_0's auc: 0.874129	valid_0's binary_logloss: 0.124312	valid_1's auc: 0.839589	valid_1's binary_logloss: 0.13918
    [31]	valid_0's auc: 0.875305	valid_0's binary_logloss: 0.123988	valid_1's auc: 0.839441	valid_1's binary_logloss: 0.139184
    [32]	valid_0's auc: 0.875943	valid_0's binary_logloss: 0.123748	valid_1's auc: 0.839268	valid_1's binary_logloss: 0.13919
    [33]	valid_0's auc: 0.876575	valid_0's binary_logloss: 0.123484	valid_1's auc: 0.839549	valid_1's binary_logloss: 0.139075
    [34]	valid_0's auc: 0.877426	valid_0's binary_logloss: 0.123156	valid_1's auc: 0.839087	valid_1's binary_logloss: 0.139148
    [35]	valid_0's auc: 0.87822	valid_0's binary_logloss: 0.122873	valid_1's auc: 0.8389	valid_1's binary_logloss: 0.139187
    [36]	valid_0's auc: 0.878932	valid_0's binary_logloss: 0.12259	valid_1's auc: 0.838921	valid_1's binary_logloss: 0.139194
    [37]	valid_0's auc: 0.879842	valid_0's binary_logloss: 0.12233	valid_1's auc: 0.839133	valid_1's binary_logloss: 0.139161
    [38]	valid_0's auc: 0.880497	valid_0's binary_logloss: 0.12208	valid_1's auc: 0.838975	valid_1's binary_logloss: 0.139143
    [39]	valid_0's auc: 0.881056	valid_0's binary_logloss: 0.121827	valid_1's auc: 0.839037	valid_1's binary_logloss: 0.139138
    [40]	valid_0's auc: 0.881604	valid_0's binary_logloss: 0.121603	valid_1's auc: 0.839204	valid_1's binary_logloss: 0.139119
    [41]	valid_0's auc: 0.882159	valid_0's binary_logloss: 0.121355	valid_1's auc: 0.839277	valid_1's binary_logloss: 0.139091
    [42]	valid_0's auc: 0.882757	valid_0's binary_logloss: 0.121116	valid_1's auc: 0.838964	valid_1's binary_logloss: 0.139133
    [43]	valid_0's auc: 0.883143	valid_0's binary_logloss: 0.120918	valid_1's auc: 0.839024	valid_1's binary_logloss: 0.139124
    [44]	valid_0's auc: 0.883697	valid_0's binary_logloss: 0.12072	valid_1's auc: 0.838652	valid_1's binary_logloss: 0.139203
    [45]	valid_0's auc: 0.884292	valid_0's binary_logloss: 0.120482	valid_1's auc: 0.839016	valid_1's binary_logloss: 0.139124
    [46]	valid_0's auc: 0.884969	valid_0's binary_logloss: 0.120266	valid_1's auc: 0.838683	valid_1's binary_logloss: 0.139184
    [47]	valid_0's auc: 0.8853	valid_0's binary_logloss: 0.120089	valid_1's auc: 0.838624	valid_1's binary_logloss: 0.139193
    [48]	valid_0's auc: 0.885876	valid_0's binary_logloss: 0.11993	valid_1's auc: 0.838569	valid_1's binary_logloss: 0.139212
    [49]	valid_0's auc: 0.886141	valid_0's binary_logloss: 0.119757	valid_1's auc: 0.838345	valid_1's binary_logloss: 0.139288
    [50]	valid_0's auc: 0.886433	valid_0's binary_logloss: 0.119595	valid_1's auc: 0.838342	valid_1's binary_logloss: 0.139332
    [51]	valid_0's auc: 0.886975	valid_0's binary_logloss: 0.119377	valid_1's auc: 0.838335	valid_1's binary_logloss: 0.139331
    [52]	valid_0's auc: 0.887568	valid_0's binary_logloss: 0.119161	valid_1's auc: 0.838204	valid_1's binary_logloss: 0.139331
    [53]	valid_0's auc: 0.887867	valid_0's binary_logloss: 0.118974	valid_1's auc: 0.838044	valid_1's binary_logloss: 0.13936
    [54]	valid_0's auc: 0.888093	valid_0's binary_logloss: 0.118834	valid_1's auc: 0.838137	valid_1's binary_logloss: 0.13935
    [55]	valid_0's auc: 0.888289	valid_0's binary_logloss: 0.118675	valid_1's auc: 0.837878	valid_1's binary_logloss: 0.139392
    [56]	valid_0's auc: 0.888615	valid_0's binary_logloss: 0.118561	valid_1's auc: 0.837776	valid_1's binary_logloss: 0.139418
    [57]	valid_0's auc: 0.889157	valid_0's binary_logloss: 0.118369	valid_1's auc: 0.837585	valid_1's binary_logloss: 0.139447
    [58]	valid_0's auc: 0.889659	valid_0's binary_logloss: 0.11819	valid_1's auc: 0.837789	valid_1's binary_logloss: 0.139431
    Early stopping, best iteration is:
    [28]	valid_0's auc: 0.87235	valid_0's binary_logloss: 0.12484	valid_1's auc: 0.840114	valid_1's binary_logloss: 0.139236
    [1]	valid_0's auc: 0.832891	valid_0's binary_logloss: 0.155302	valid_1's auc: 0.818851	valid_1's binary_logloss: 0.164826
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.84519	valid_0's binary_logloss: 0.149727	valid_1's auc: 0.827144	valid_1's binary_logloss: 0.159879
    [3]	valid_0's auc: 0.848018	valid_0's binary_logloss: 0.145627	valid_1's auc: 0.826851	valid_1's binary_logloss: 0.15631
    [4]	valid_0's auc: 0.851096	valid_0's binary_logloss: 0.142423	valid_1's auc: 0.83073	valid_1's binary_logloss: 0.1534
    [5]	valid_0's auc: 0.854735	valid_0's binary_logloss: 0.139746	valid_1's auc: 0.832753	valid_1's binary_logloss: 0.151136
    [6]	valid_0's auc: 0.856928	valid_0's binary_logloss: 0.137509	valid_1's auc: 0.835605	valid_1's binary_logloss: 0.14924
    [7]	valid_0's auc: 0.859448	valid_0's binary_logloss: 0.135575	valid_1's auc: 0.835612	valid_1's binary_logloss: 0.147799
    [8]	valid_0's auc: 0.861685	valid_0's binary_logloss: 0.133953	valid_1's auc: 0.834408	valid_1's binary_logloss: 0.146634
    [9]	valid_0's auc: 0.863391	valid_0's binary_logloss: 0.132468	valid_1's auc: 0.835623	valid_1's binary_logloss: 0.145549
    [10]	valid_0's auc: 0.865858	valid_0's binary_logloss: 0.131185	valid_1's auc: 0.83487	valid_1's binary_logloss: 0.144745
    [11]	valid_0's auc: 0.867134	valid_0's binary_logloss: 0.130116	valid_1's auc: 0.834692	valid_1's binary_logloss: 0.14411
    [12]	valid_0's auc: 0.868217	valid_0's binary_logloss: 0.129097	valid_1's auc: 0.834746	valid_1's binary_logloss: 0.143527
    [13]	valid_0's auc: 0.87073	valid_0's binary_logloss: 0.128129	valid_1's auc: 0.833582	valid_1's binary_logloss: 0.143122
    [14]	valid_0's auc: 0.872621	valid_0's binary_logloss: 0.12721	valid_1's auc: 0.833205	valid_1's binary_logloss: 0.142745
    [15]	valid_0's auc: 0.874007	valid_0's binary_logloss: 0.126363	valid_1's auc: 0.83246	valid_1's binary_logloss: 0.142489
    [16]	valid_0's auc: 0.875141	valid_0's binary_logloss: 0.125606	valid_1's auc: 0.831958	valid_1's binary_logloss: 0.142275
    [17]	valid_0's auc: 0.876061	valid_0's binary_logloss: 0.124928	valid_1's auc: 0.831586	valid_1's binary_logloss: 0.142141
    [18]	valid_0's auc: 0.876982	valid_0's binary_logloss: 0.124313	valid_1's auc: 0.830954	valid_1's binary_logloss: 0.142066
    [19]	valid_0's auc: 0.877885	valid_0's binary_logloss: 0.123709	valid_1's auc: 0.830572	valid_1's binary_logloss: 0.14196
    [20]	valid_0's auc: 0.879378	valid_0's binary_logloss: 0.123088	valid_1's auc: 0.830076	valid_1's binary_logloss: 0.14196
    [21]	valid_0's auc: 0.880647	valid_0's binary_logloss: 0.122488	valid_1's auc: 0.830109	valid_1's binary_logloss: 0.141858
    [22]	valid_0's auc: 0.881614	valid_0's binary_logloss: 0.121973	valid_1's auc: 0.829735	valid_1's binary_logloss: 0.141822
    [23]	valid_0's auc: 0.882402	valid_0's binary_logloss: 0.121554	valid_1's auc: 0.829254	valid_1's binary_logloss: 0.141805
    [24]	valid_0's auc: 0.883011	valid_0's binary_logloss: 0.121078	valid_1's auc: 0.829054	valid_1's binary_logloss: 0.14178
    [25]	valid_0's auc: 0.884627	valid_0's binary_logloss: 0.120587	valid_1's auc: 0.82942	valid_1's binary_logloss: 0.141653
    [26]	valid_0's auc: 0.885304	valid_0's binary_logloss: 0.120169	valid_1's auc: 0.828716	valid_1's binary_logloss: 0.141755
    [27]	valid_0's auc: 0.88664	valid_0's binary_logloss: 0.119673	valid_1's auc: 0.828869	valid_1's binary_logloss: 0.141682
    [28]	valid_0's auc: 0.887143	valid_0's binary_logloss: 0.119308	valid_1's auc: 0.828987	valid_1's binary_logloss: 0.141649
    [29]	valid_0's auc: 0.88825	valid_0's binary_logloss: 0.1189	valid_1's auc: 0.829075	valid_1's binary_logloss: 0.141601
    [30]	valid_0's auc: 0.889081	valid_0's binary_logloss: 0.118531	valid_1's auc: 0.828871	valid_1's binary_logloss: 0.141605
    [31]	valid_0's auc: 0.890195	valid_0's binary_logloss: 0.118117	valid_1's auc: 0.828972	valid_1's binary_logloss: 0.141605
    [32]	valid_0's auc: 0.890928	valid_0's binary_logloss: 0.117735	valid_1's auc: 0.827969	valid_1's binary_logloss: 0.141796
    [33]	valid_0's auc: 0.891505	valid_0's binary_logloss: 0.117389	valid_1's auc: 0.827611	valid_1's binary_logloss: 0.141916
    [34]	valid_0's auc: 0.892223	valid_0's binary_logloss: 0.11707	valid_1's auc: 0.827019	valid_1's binary_logloss: 0.142051
    [35]	valid_0's auc: 0.892825	valid_0's binary_logloss: 0.116751	valid_1's auc: 0.826865	valid_1's binary_logloss: 0.142116
    [36]	valid_0's auc: 0.893984	valid_0's binary_logloss: 0.116353	valid_1's auc: 0.827203	valid_1's binary_logloss: 0.14207
    [37]	valid_0's auc: 0.89456	valid_0's binary_logloss: 0.11603	valid_1's auc: 0.827292	valid_1's binary_logloss: 0.142005
    [38]	valid_0's auc: 0.89511	valid_0's binary_logloss: 0.115713	valid_1's auc: 0.827214	valid_1's binary_logloss: 0.14206
    [39]	valid_0's auc: 0.895738	valid_0's binary_logloss: 0.115415	valid_1's auc: 0.82695	valid_1's binary_logloss: 0.142162
    Early stopping, best iteration is:
    [9]	valid_0's auc: 0.863391	valid_0's binary_logloss: 0.132468	valid_1's auc: 0.835623	valid_1's binary_logloss: 0.145549
    [1]	valid_0's auc: 0.833054	valid_0's binary_logloss: 0.15572	valid_1's auc: 0.817048	valid_1's binary_logloss: 0.165036
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.841397	valid_0's binary_logloss: 0.149862	valid_1's auc: 0.82157	valid_1's binary_logloss: 0.159575
    [3]	valid_0's auc: 0.849058	valid_0's binary_logloss: 0.145662	valid_1's auc: 0.829866	valid_1's binary_logloss: 0.155774
    [4]	valid_0's auc: 0.854301	valid_0's binary_logloss: 0.142356	valid_1's auc: 0.832415	valid_1's binary_logloss: 0.152936
    [5]	valid_0's auc: 0.858045	valid_0's binary_logloss: 0.139697	valid_1's auc: 0.834554	valid_1's binary_logloss: 0.150635
    [6]	valid_0's auc: 0.860767	valid_0's binary_logloss: 0.137458	valid_1's auc: 0.834885	valid_1's binary_logloss: 0.148761
    [7]	valid_0's auc: 0.863011	valid_0's binary_logloss: 0.135522	valid_1's auc: 0.835812	valid_1's binary_logloss: 0.147245
    [8]	valid_0's auc: 0.864923	valid_0's binary_logloss: 0.133792	valid_1's auc: 0.836656	valid_1's binary_logloss: 0.145923
    [9]	valid_0's auc: 0.865706	valid_0's binary_logloss: 0.13236	valid_1's auc: 0.836912	valid_1's binary_logloss: 0.144867
    [10]	valid_0's auc: 0.867693	valid_0's binary_logloss: 0.131066	valid_1's auc: 0.837266	valid_1's binary_logloss: 0.143895
    [11]	valid_0's auc: 0.868596	valid_0's binary_logloss: 0.129937	valid_1's auc: 0.836466	valid_1's binary_logloss: 0.143255
    [12]	valid_0's auc: 0.87012	valid_0's binary_logloss: 0.128904	valid_1's auc: 0.836589	valid_1's binary_logloss: 0.142728
    [13]	valid_0's auc: 0.871703	valid_0's binary_logloss: 0.127913	valid_1's auc: 0.836567	valid_1's binary_logloss: 0.142105
    [14]	valid_0's auc: 0.873468	valid_0's binary_logloss: 0.126983	valid_1's auc: 0.835538	valid_1's binary_logloss: 0.141771
    [15]	valid_0's auc: 0.874839	valid_0's binary_logloss: 0.126147	valid_1's auc: 0.835363	valid_1's binary_logloss: 0.141464
    [16]	valid_0's auc: 0.876399	valid_0's binary_logloss: 0.125331	valid_1's auc: 0.83478	valid_1's binary_logloss: 0.141245
    [17]	valid_0's auc: 0.877465	valid_0's binary_logloss: 0.124655	valid_1's auc: 0.834621	valid_1's binary_logloss: 0.141028
    [18]	valid_0's auc: 0.878935	valid_0's binary_logloss: 0.123944	valid_1's auc: 0.834165	valid_1's binary_logloss: 0.140935
    [19]	valid_0's auc: 0.88046	valid_0's binary_logloss: 0.123313	valid_1's auc: 0.834629	valid_1's binary_logloss: 0.140738
    [20]	valid_0's auc: 0.881517	valid_0's binary_logloss: 0.12269	valid_1's auc: 0.8347	valid_1's binary_logloss: 0.140611
    [21]	valid_0's auc: 0.882464	valid_0's binary_logloss: 0.122095	valid_1's auc: 0.834656	valid_1's binary_logloss: 0.140487
    [22]	valid_0's auc: 0.883744	valid_0's binary_logloss: 0.121504	valid_1's auc: 0.834562	valid_1's binary_logloss: 0.140328
    [23]	valid_0's auc: 0.885301	valid_0's binary_logloss: 0.12091	valid_1's auc: 0.835278	valid_1's binary_logloss: 0.140199
    [24]	valid_0's auc: 0.886266	valid_0's binary_logloss: 0.120437	valid_1's auc: 0.835728	valid_1's binary_logloss: 0.140094
    [25]	valid_0's auc: 0.88755	valid_0's binary_logloss: 0.119931	valid_1's auc: 0.836199	valid_1's binary_logloss: 0.140076
    [26]	valid_0's auc: 0.888525	valid_0's binary_logloss: 0.119473	valid_1's auc: 0.836708	valid_1's binary_logloss: 0.139945
    [27]	valid_0's auc: 0.889589	valid_0's binary_logloss: 0.119012	valid_1's auc: 0.836951	valid_1's binary_logloss: 0.139843
    [28]	valid_0's auc: 0.890552	valid_0's binary_logloss: 0.118602	valid_1's auc: 0.836524	valid_1's binary_logloss: 0.139871
    [29]	valid_0's auc: 0.891402	valid_0's binary_logloss: 0.118166	valid_1's auc: 0.836264	valid_1's binary_logloss: 0.139884
    [30]	valid_0's auc: 0.891982	valid_0's binary_logloss: 0.117805	valid_1's auc: 0.835959	valid_1's binary_logloss: 0.139937
    [31]	valid_0's auc: 0.893185	valid_0's binary_logloss: 0.117392	valid_1's auc: 0.836384	valid_1's binary_logloss: 0.13992
    [32]	valid_0's auc: 0.894065	valid_0's binary_logloss: 0.117017	valid_1's auc: 0.836341	valid_1's binary_logloss: 0.139888
    [33]	valid_0's auc: 0.894791	valid_0's binary_logloss: 0.116671	valid_1's auc: 0.836753	valid_1's binary_logloss: 0.139812
    [34]	valid_0's auc: 0.895313	valid_0's binary_logloss: 0.116321	valid_1's auc: 0.836733	valid_1's binary_logloss: 0.139826
    [35]	valid_0's auc: 0.895876	valid_0's binary_logloss: 0.116039	valid_1's auc: 0.836245	valid_1's binary_logloss: 0.139883
    [36]	valid_0's auc: 0.896909	valid_0's binary_logloss: 0.115684	valid_1's auc: 0.836079	valid_1's binary_logloss: 0.139912
    [37]	valid_0's auc: 0.897427	valid_0's binary_logloss: 0.115388	valid_1's auc: 0.835564	valid_1's binary_logloss: 0.140024
    [38]	valid_0's auc: 0.898442	valid_0's binary_logloss: 0.115006	valid_1's auc: 0.835612	valid_1's binary_logloss: 0.140075
    [39]	valid_0's auc: 0.899304	valid_0's binary_logloss: 0.114592	valid_1's auc: 0.836273	valid_1's binary_logloss: 0.139974
    [40]	valid_0's auc: 0.89974	valid_0's binary_logloss: 0.11432	valid_1's auc: 0.836096	valid_1's binary_logloss: 0.140042
    Early stopping, best iteration is:
    [10]	valid_0's auc: 0.867693	valid_0's binary_logloss: 0.131066	valid_1's auc: 0.837266	valid_1's binary_logloss: 0.143895
    [1]	valid_0's auc: 0.830643	valid_0's binary_logloss: 0.155759	valid_1's auc: 0.816734	valid_1's binary_logloss: 0.164985
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.839353	valid_0's binary_logloss: 0.149977	valid_1's auc: 0.822571	valid_1's binary_logloss: 0.159808
    [3]	valid_0's auc: 0.847366	valid_0's binary_logloss: 0.145866	valid_1's auc: 0.829312	valid_1's binary_logloss: 0.156171
    [4]	valid_0's auc: 0.850911	valid_0's binary_logloss: 0.14247	valid_1's auc: 0.830848	valid_1's binary_logloss: 0.153328
    [5]	valid_0's auc: 0.854674	valid_0's binary_logloss: 0.139764	valid_1's auc: 0.833041	valid_1's binary_logloss: 0.151023
    [6]	valid_0's auc: 0.856722	valid_0's binary_logloss: 0.1375	valid_1's auc: 0.834264	valid_1's binary_logloss: 0.149166
    [7]	valid_0's auc: 0.858253	valid_0's binary_logloss: 0.135713	valid_1's auc: 0.834998	valid_1's binary_logloss: 0.147631
    [8]	valid_0's auc: 0.859768	valid_0's binary_logloss: 0.134063	valid_1's auc: 0.835678	valid_1's binary_logloss: 0.146384
    [9]	valid_0's auc: 0.86262	valid_0's binary_logloss: 0.132622	valid_1's auc: 0.836272	valid_1's binary_logloss: 0.145313
    [10]	valid_0's auc: 0.864631	valid_0's binary_logloss: 0.131324	valid_1's auc: 0.835827	valid_1's binary_logloss: 0.144553
    [11]	valid_0's auc: 0.866805	valid_0's binary_logloss: 0.130172	valid_1's auc: 0.835375	valid_1's binary_logloss: 0.143933
    [12]	valid_0's auc: 0.868266	valid_0's binary_logloss: 0.129101	valid_1's auc: 0.835951	valid_1's binary_logloss: 0.143342
    [13]	valid_0's auc: 0.870762	valid_0's binary_logloss: 0.128144	valid_1's auc: 0.83626	valid_1's binary_logloss: 0.142813
    [14]	valid_0's auc: 0.872747	valid_0's binary_logloss: 0.127222	valid_1's auc: 0.835864	valid_1's binary_logloss: 0.142466
    [15]	valid_0's auc: 0.874158	valid_0's binary_logloss: 0.126428	valid_1's auc: 0.83548	valid_1's binary_logloss: 0.142108
    [16]	valid_0's auc: 0.875931	valid_0's binary_logloss: 0.125651	valid_1's auc: 0.836367	valid_1's binary_logloss: 0.141684
    [17]	valid_0's auc: 0.876854	valid_0's binary_logloss: 0.124918	valid_1's auc: 0.835689	valid_1's binary_logloss: 0.141524
    [18]	valid_0's auc: 0.878211	valid_0's binary_logloss: 0.124197	valid_1's auc: 0.835893	valid_1's binary_logloss: 0.141285
    [19]	valid_0's auc: 0.879125	valid_0's binary_logloss: 0.123553	valid_1's auc: 0.835877	valid_1's binary_logloss: 0.141128
    [20]	valid_0's auc: 0.880489	valid_0's binary_logloss: 0.122856	valid_1's auc: 0.835385	valid_1's binary_logloss: 0.141032
    [21]	valid_0's auc: 0.881696	valid_0's binary_logloss: 0.122219	valid_1's auc: 0.835822	valid_1's binary_logloss: 0.140843
    [22]	valid_0's auc: 0.882257	valid_0's binary_logloss: 0.121726	valid_1's auc: 0.835849	valid_1's binary_logloss: 0.140761
    [23]	valid_0's auc: 0.883635	valid_0's binary_logloss: 0.121206	valid_1's auc: 0.836223	valid_1's binary_logloss: 0.140607
    [24]	valid_0's auc: 0.884533	valid_0's binary_logloss: 0.120734	valid_1's auc: 0.836473	valid_1's binary_logloss: 0.14049
    [25]	valid_0's auc: 0.885234	valid_0's binary_logloss: 0.120268	valid_1's auc: 0.836722	valid_1's binary_logloss: 0.140403
    [26]	valid_0's auc: 0.886292	valid_0's binary_logloss: 0.119794	valid_1's auc: 0.836549	valid_1's binary_logloss: 0.140423
    [27]	valid_0's auc: 0.887064	valid_0's binary_logloss: 0.119366	valid_1's auc: 0.836155	valid_1's binary_logloss: 0.140447
    [28]	valid_0's auc: 0.887621	valid_0's binary_logloss: 0.119008	valid_1's auc: 0.835594	valid_1's binary_logloss: 0.140532
    [29]	valid_0's auc: 0.888965	valid_0's binary_logloss: 0.118547	valid_1's auc: 0.835464	valid_1's binary_logloss: 0.140508
    [30]	valid_0's auc: 0.889898	valid_0's binary_logloss: 0.118139	valid_1's auc: 0.83577	valid_1's binary_logloss: 0.140461
    [31]	valid_0's auc: 0.890896	valid_0's binary_logloss: 0.117734	valid_1's auc: 0.835475	valid_1's binary_logloss: 0.140463
    [32]	valid_0's auc: 0.892374	valid_0's binary_logloss: 0.1173	valid_1's auc: 0.835364	valid_1's binary_logloss: 0.140506
    [33]	valid_0's auc: 0.893164	valid_0's binary_logloss: 0.116978	valid_1's auc: 0.835865	valid_1's binary_logloss: 0.14041
    [34]	valid_0's auc: 0.893848	valid_0's binary_logloss: 0.11662	valid_1's auc: 0.836021	valid_1's binary_logloss: 0.140353
    [35]	valid_0's auc: 0.894232	valid_0's binary_logloss: 0.116323	valid_1's auc: 0.8359	valid_1's binary_logloss: 0.140396
    [36]	valid_0's auc: 0.895003	valid_0's binary_logloss: 0.115986	valid_1's auc: 0.835855	valid_1's binary_logloss: 0.140416
    [37]	valid_0's auc: 0.895898	valid_0's binary_logloss: 0.115609	valid_1's auc: 0.836185	valid_1's binary_logloss: 0.140369
    [38]	valid_0's auc: 0.896459	valid_0's binary_logloss: 0.11527	valid_1's auc: 0.835754	valid_1's binary_logloss: 0.140443
    [39]	valid_0's auc: 0.897377	valid_0's binary_logloss: 0.114873	valid_1's auc: 0.835638	valid_1's binary_logloss: 0.140474
    [40]	valid_0's auc: 0.89776	valid_0's binary_logloss: 0.114588	valid_1's auc: 0.835639	valid_1's binary_logloss: 0.140491
    [41]	valid_0's auc: 0.898583	valid_0's binary_logloss: 0.114302	valid_1's auc: 0.835705	valid_1's binary_logloss: 0.140506
    [42]	valid_0's auc: 0.899197	valid_0's binary_logloss: 0.113975	valid_1's auc: 0.835052	valid_1's binary_logloss: 0.14064
    [43]	valid_0's auc: 0.899803	valid_0's binary_logloss: 0.113654	valid_1's auc: 0.835035	valid_1's binary_logloss: 0.140691
    [44]	valid_0's auc: 0.900641	valid_0's binary_logloss: 0.113388	valid_1's auc: 0.835214	valid_1's binary_logloss: 0.140703
    [45]	valid_0's auc: 0.900962	valid_0's binary_logloss: 0.113098	valid_1's auc: 0.835276	valid_1's binary_logloss: 0.140695
    [46]	valid_0's auc: 0.901584	valid_0's binary_logloss: 0.112771	valid_1's auc: 0.83495	valid_1's binary_logloss: 0.140754
    [47]	valid_0's auc: 0.902256	valid_0's binary_logloss: 0.112493	valid_1's auc: 0.835639	valid_1's binary_logloss: 0.14064
    [48]	valid_0's auc: 0.902688	valid_0's binary_logloss: 0.112198	valid_1's auc: 0.835495	valid_1's binary_logloss: 0.140691
    [49]	valid_0's auc: 0.902922	valid_0's binary_logloss: 0.111944	valid_1's auc: 0.835281	valid_1's binary_logloss: 0.140819
    [50]	valid_0's auc: 0.903747	valid_0's binary_logloss: 0.111595	valid_1's auc: 0.835359	valid_1's binary_logloss: 0.140811
    [51]	valid_0's auc: 0.904427	valid_0's binary_logloss: 0.111354	valid_1's auc: 0.835245	valid_1's binary_logloss: 0.140873
    [52]	valid_0's auc: 0.90467	valid_0's binary_logloss: 0.111111	valid_1's auc: 0.835057	valid_1's binary_logloss: 0.140993
    [53]	valid_0's auc: 0.904868	valid_0's binary_logloss: 0.110853	valid_1's auc: 0.834751	valid_1's binary_logloss: 0.14108
    [54]	valid_0's auc: 0.905166	valid_0's binary_logloss: 0.110627	valid_1's auc: 0.83411	valid_1's binary_logloss: 0.141282
    [55]	valid_0's auc: 0.905665	valid_0's binary_logloss: 0.110375	valid_1's auc: 0.833739	valid_1's binary_logloss: 0.141413
    Early stopping, best iteration is:
    [25]	valid_0's auc: 0.885234	valid_0's binary_logloss: 0.120268	valid_1's auc: 0.836722	valid_1's binary_logloss: 0.140403
    [1]	valid_0's auc: 0.832891	valid_0's binary_logloss: 0.155302	valid_1's auc: 0.818851	valid_1's binary_logloss: 0.164826
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.84519	valid_0's binary_logloss: 0.149727	valid_1's auc: 0.827144	valid_1's binary_logloss: 0.159879
    [3]	valid_0's auc: 0.848018	valid_0's binary_logloss: 0.145627	valid_1's auc: 0.826851	valid_1's binary_logloss: 0.15631
    [4]	valid_0's auc: 0.851096	valid_0's binary_logloss: 0.142423	valid_1's auc: 0.83073	valid_1's binary_logloss: 0.1534
    [5]	valid_0's auc: 0.854735	valid_0's binary_logloss: 0.139746	valid_1's auc: 0.832753	valid_1's binary_logloss: 0.151136
    [6]	valid_0's auc: 0.856928	valid_0's binary_logloss: 0.137509	valid_1's auc: 0.835605	valid_1's binary_logloss: 0.14924
    [7]	valid_0's auc: 0.859448	valid_0's binary_logloss: 0.135575	valid_1's auc: 0.835612	valid_1's binary_logloss: 0.147799
    [8]	valid_0's auc: 0.861685	valid_0's binary_logloss: 0.133953	valid_1's auc: 0.834408	valid_1's binary_logloss: 0.146634
    [9]	valid_0's auc: 0.863391	valid_0's binary_logloss: 0.132468	valid_1's auc: 0.835623	valid_1's binary_logloss: 0.145549
    [10]	valid_0's auc: 0.865858	valid_0's binary_logloss: 0.131185	valid_1's auc: 0.83487	valid_1's binary_logloss: 0.144745
    [11]	valid_0's auc: 0.867134	valid_0's binary_logloss: 0.130116	valid_1's auc: 0.834692	valid_1's binary_logloss: 0.14411
    [12]	valid_0's auc: 0.868217	valid_0's binary_logloss: 0.129097	valid_1's auc: 0.834746	valid_1's binary_logloss: 0.143527
    [13]	valid_0's auc: 0.87073	valid_0's binary_logloss: 0.128129	valid_1's auc: 0.833582	valid_1's binary_logloss: 0.143122
    [14]	valid_0's auc: 0.872621	valid_0's binary_logloss: 0.12721	valid_1's auc: 0.833205	valid_1's binary_logloss: 0.142745
    [15]	valid_0's auc: 0.874007	valid_0's binary_logloss: 0.126363	valid_1's auc: 0.83246	valid_1's binary_logloss: 0.142489
    [16]	valid_0's auc: 0.875141	valid_0's binary_logloss: 0.125606	valid_1's auc: 0.831958	valid_1's binary_logloss: 0.142275
    [17]	valid_0's auc: 0.876061	valid_0's binary_logloss: 0.124928	valid_1's auc: 0.831586	valid_1's binary_logloss: 0.142141
    [18]	valid_0's auc: 0.876982	valid_0's binary_logloss: 0.124313	valid_1's auc: 0.830954	valid_1's binary_logloss: 0.142066
    [19]	valid_0's auc: 0.877885	valid_0's binary_logloss: 0.123709	valid_1's auc: 0.830572	valid_1's binary_logloss: 0.14196
    [20]	valid_0's auc: 0.879378	valid_0's binary_logloss: 0.123088	valid_1's auc: 0.830076	valid_1's binary_logloss: 0.14196
    [21]	valid_0's auc: 0.880647	valid_0's binary_logloss: 0.122488	valid_1's auc: 0.830109	valid_1's binary_logloss: 0.141858
    [22]	valid_0's auc: 0.881614	valid_0's binary_logloss: 0.121973	valid_1's auc: 0.829735	valid_1's binary_logloss: 0.141822
    [23]	valid_0's auc: 0.882402	valid_0's binary_logloss: 0.121554	valid_1's auc: 0.829254	valid_1's binary_logloss: 0.141805
    [24]	valid_0's auc: 0.883011	valid_0's binary_logloss: 0.121078	valid_1's auc: 0.829054	valid_1's binary_logloss: 0.14178
    [25]	valid_0's auc: 0.884627	valid_0's binary_logloss: 0.120587	valid_1's auc: 0.82942	valid_1's binary_logloss: 0.141653
    [26]	valid_0's auc: 0.885304	valid_0's binary_logloss: 0.120169	valid_1's auc: 0.828716	valid_1's binary_logloss: 0.141755
    [27]	valid_0's auc: 0.88664	valid_0's binary_logloss: 0.119673	valid_1's auc: 0.828869	valid_1's binary_logloss: 0.141682
    [28]	valid_0's auc: 0.887143	valid_0's binary_logloss: 0.119308	valid_1's auc: 0.828987	valid_1's binary_logloss: 0.141649
    [29]	valid_0's auc: 0.88825	valid_0's binary_logloss: 0.1189	valid_1's auc: 0.829075	valid_1's binary_logloss: 0.141601
    [30]	valid_0's auc: 0.889081	valid_0's binary_logloss: 0.118531	valid_1's auc: 0.828871	valid_1's binary_logloss: 0.141605
    [31]	valid_0's auc: 0.890195	valid_0's binary_logloss: 0.118117	valid_1's auc: 0.828972	valid_1's binary_logloss: 0.141605
    [32]	valid_0's auc: 0.890928	valid_0's binary_logloss: 0.117735	valid_1's auc: 0.827969	valid_1's binary_logloss: 0.141796
    [33]	valid_0's auc: 0.891505	valid_0's binary_logloss: 0.117389	valid_1's auc: 0.827611	valid_1's binary_logloss: 0.141916
    [34]	valid_0's auc: 0.892223	valid_0's binary_logloss: 0.11707	valid_1's auc: 0.827019	valid_1's binary_logloss: 0.142051
    [35]	valid_0's auc: 0.892825	valid_0's binary_logloss: 0.116751	valid_1's auc: 0.826865	valid_1's binary_logloss: 0.142116
    [36]	valid_0's auc: 0.893984	valid_0's binary_logloss: 0.116353	valid_1's auc: 0.827203	valid_1's binary_logloss: 0.14207
    [37]	valid_0's auc: 0.89456	valid_0's binary_logloss: 0.11603	valid_1's auc: 0.827292	valid_1's binary_logloss: 0.142005
    [38]	valid_0's auc: 0.89511	valid_0's binary_logloss: 0.115713	valid_1's auc: 0.827214	valid_1's binary_logloss: 0.14206
    [39]	valid_0's auc: 0.895738	valid_0's binary_logloss: 0.115415	valid_1's auc: 0.82695	valid_1's binary_logloss: 0.142162
    Early stopping, best iteration is:
    [9]	valid_0's auc: 0.863391	valid_0's binary_logloss: 0.132468	valid_1's auc: 0.835623	valid_1's binary_logloss: 0.145549
    [1]	valid_0's auc: 0.833054	valid_0's binary_logloss: 0.15572	valid_1's auc: 0.817048	valid_1's binary_logloss: 0.165036
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.841397	valid_0's binary_logloss: 0.149862	valid_1's auc: 0.82157	valid_1's binary_logloss: 0.159575
    [3]	valid_0's auc: 0.849058	valid_0's binary_logloss: 0.145662	valid_1's auc: 0.829866	valid_1's binary_logloss: 0.155774
    [4]	valid_0's auc: 0.854301	valid_0's binary_logloss: 0.142356	valid_1's auc: 0.832415	valid_1's binary_logloss: 0.152936
    [5]	valid_0's auc: 0.858045	valid_0's binary_logloss: 0.139697	valid_1's auc: 0.834554	valid_1's binary_logloss: 0.150635
    [6]	valid_0's auc: 0.860767	valid_0's binary_logloss: 0.137458	valid_1's auc: 0.834885	valid_1's binary_logloss: 0.148761
    [7]	valid_0's auc: 0.863011	valid_0's binary_logloss: 0.135522	valid_1's auc: 0.835812	valid_1's binary_logloss: 0.147245
    [8]	valid_0's auc: 0.864923	valid_0's binary_logloss: 0.133792	valid_1's auc: 0.836656	valid_1's binary_logloss: 0.145923
    [9]	valid_0's auc: 0.865706	valid_0's binary_logloss: 0.13236	valid_1's auc: 0.836912	valid_1's binary_logloss: 0.144867
    [10]	valid_0's auc: 0.867693	valid_0's binary_logloss: 0.131066	valid_1's auc: 0.837266	valid_1's binary_logloss: 0.143895
    [11]	valid_0's auc: 0.868596	valid_0's binary_logloss: 0.129937	valid_1's auc: 0.836466	valid_1's binary_logloss: 0.143255
    [12]	valid_0's auc: 0.87012	valid_0's binary_logloss: 0.128904	valid_1's auc: 0.836589	valid_1's binary_logloss: 0.142728
    [13]	valid_0's auc: 0.871703	valid_0's binary_logloss: 0.127913	valid_1's auc: 0.836567	valid_1's binary_logloss: 0.142105
    [14]	valid_0's auc: 0.873468	valid_0's binary_logloss: 0.126983	valid_1's auc: 0.835538	valid_1's binary_logloss: 0.141771
    [15]	valid_0's auc: 0.874839	valid_0's binary_logloss: 0.126147	valid_1's auc: 0.835363	valid_1's binary_logloss: 0.141464
    [16]	valid_0's auc: 0.876399	valid_0's binary_logloss: 0.125331	valid_1's auc: 0.83478	valid_1's binary_logloss: 0.141245
    [17]	valid_0's auc: 0.877465	valid_0's binary_logloss: 0.124655	valid_1's auc: 0.834621	valid_1's binary_logloss: 0.141028
    [18]	valid_0's auc: 0.878935	valid_0's binary_logloss: 0.123944	valid_1's auc: 0.834165	valid_1's binary_logloss: 0.140935
    [19]	valid_0's auc: 0.88046	valid_0's binary_logloss: 0.123313	valid_1's auc: 0.834629	valid_1's binary_logloss: 0.140738
    [20]	valid_0's auc: 0.881517	valid_0's binary_logloss: 0.12269	valid_1's auc: 0.8347	valid_1's binary_logloss: 0.140611
    [21]	valid_0's auc: 0.882464	valid_0's binary_logloss: 0.122095	valid_1's auc: 0.834656	valid_1's binary_logloss: 0.140487
    [22]	valid_0's auc: 0.883744	valid_0's binary_logloss: 0.121504	valid_1's auc: 0.834562	valid_1's binary_logloss: 0.140328
    [23]	valid_0's auc: 0.885301	valid_0's binary_logloss: 0.12091	valid_1's auc: 0.835278	valid_1's binary_logloss: 0.140199
    [24]	valid_0's auc: 0.886266	valid_0's binary_logloss: 0.120437	valid_1's auc: 0.835728	valid_1's binary_logloss: 0.140094
    [25]	valid_0's auc: 0.88755	valid_0's binary_logloss: 0.119931	valid_1's auc: 0.836199	valid_1's binary_logloss: 0.140076
    [26]	valid_0's auc: 0.888525	valid_0's binary_logloss: 0.119473	valid_1's auc: 0.836708	valid_1's binary_logloss: 0.139945
    [27]	valid_0's auc: 0.889589	valid_0's binary_logloss: 0.119012	valid_1's auc: 0.836951	valid_1's binary_logloss: 0.139843
    [28]	valid_0's auc: 0.890552	valid_0's binary_logloss: 0.118602	valid_1's auc: 0.836524	valid_1's binary_logloss: 0.139871
    [29]	valid_0's auc: 0.891402	valid_0's binary_logloss: 0.118166	valid_1's auc: 0.836264	valid_1's binary_logloss: 0.139884
    [30]	valid_0's auc: 0.891982	valid_0's binary_logloss: 0.117805	valid_1's auc: 0.835959	valid_1's binary_logloss: 0.139937
    [31]	valid_0's auc: 0.893185	valid_0's binary_logloss: 0.117392	valid_1's auc: 0.836384	valid_1's binary_logloss: 0.13992
    [32]	valid_0's auc: 0.894065	valid_0's binary_logloss: 0.117017	valid_1's auc: 0.836341	valid_1's binary_logloss: 0.139888
    [33]	valid_0's auc: 0.894791	valid_0's binary_logloss: 0.116671	valid_1's auc: 0.836753	valid_1's binary_logloss: 0.139812
    [34]	valid_0's auc: 0.895313	valid_0's binary_logloss: 0.116321	valid_1's auc: 0.836733	valid_1's binary_logloss: 0.139826
    [35]	valid_0's auc: 0.895876	valid_0's binary_logloss: 0.116039	valid_1's auc: 0.836245	valid_1's binary_logloss: 0.139883
    [36]	valid_0's auc: 0.896909	valid_0's binary_logloss: 0.115684	valid_1's auc: 0.836079	valid_1's binary_logloss: 0.139912
    [37]	valid_0's auc: 0.897427	valid_0's binary_logloss: 0.115388	valid_1's auc: 0.835564	valid_1's binary_logloss: 0.140024
    [38]	valid_0's auc: 0.898442	valid_0's binary_logloss: 0.115006	valid_1's auc: 0.835612	valid_1's binary_logloss: 0.140075
    [39]	valid_0's auc: 0.899304	valid_0's binary_logloss: 0.114592	valid_1's auc: 0.836273	valid_1's binary_logloss: 0.139974
    [40]	valid_0's auc: 0.89974	valid_0's binary_logloss: 0.11432	valid_1's auc: 0.836096	valid_1's binary_logloss: 0.140042
    Early stopping, best iteration is:
    [10]	valid_0's auc: 0.867693	valid_0's binary_logloss: 0.131066	valid_1's auc: 0.837266	valid_1's binary_logloss: 0.143895
    [1]	valid_0's auc: 0.830643	valid_0's binary_logloss: 0.155759	valid_1's auc: 0.816734	valid_1's binary_logloss: 0.164985
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.839353	valid_0's binary_logloss: 0.149977	valid_1's auc: 0.822571	valid_1's binary_logloss: 0.159808
    [3]	valid_0's auc: 0.847366	valid_0's binary_logloss: 0.145866	valid_1's auc: 0.829312	valid_1's binary_logloss: 0.156171
    [4]	valid_0's auc: 0.850911	valid_0's binary_logloss: 0.14247	valid_1's auc: 0.830848	valid_1's binary_logloss: 0.153328
    [5]	valid_0's auc: 0.854674	valid_0's binary_logloss: 0.139764	valid_1's auc: 0.833041	valid_1's binary_logloss: 0.151023
    [6]	valid_0's auc: 0.856722	valid_0's binary_logloss: 0.1375	valid_1's auc: 0.834264	valid_1's binary_logloss: 0.149166
    [7]	valid_0's auc: 0.858253	valid_0's binary_logloss: 0.135713	valid_1's auc: 0.834998	valid_1's binary_logloss: 0.147631
    [8]	valid_0's auc: 0.859768	valid_0's binary_logloss: 0.134063	valid_1's auc: 0.835678	valid_1's binary_logloss: 0.146384
    [9]	valid_0's auc: 0.86262	valid_0's binary_logloss: 0.132622	valid_1's auc: 0.836272	valid_1's binary_logloss: 0.145313
    [10]	valid_0's auc: 0.864631	valid_0's binary_logloss: 0.131324	valid_1's auc: 0.835827	valid_1's binary_logloss: 0.144553
    [11]	valid_0's auc: 0.866805	valid_0's binary_logloss: 0.130172	valid_1's auc: 0.835375	valid_1's binary_logloss: 0.143933
    [12]	valid_0's auc: 0.868266	valid_0's binary_logloss: 0.129101	valid_1's auc: 0.835951	valid_1's binary_logloss: 0.143342
    [13]	valid_0's auc: 0.870762	valid_0's binary_logloss: 0.128144	valid_1's auc: 0.83626	valid_1's binary_logloss: 0.142813
    [14]	valid_0's auc: 0.872747	valid_0's binary_logloss: 0.127222	valid_1's auc: 0.835864	valid_1's binary_logloss: 0.142466
    [15]	valid_0's auc: 0.874158	valid_0's binary_logloss: 0.126428	valid_1's auc: 0.83548	valid_1's binary_logloss: 0.142108
    [16]	valid_0's auc: 0.875931	valid_0's binary_logloss: 0.125651	valid_1's auc: 0.836367	valid_1's binary_logloss: 0.141684
    [17]	valid_0's auc: 0.876854	valid_0's binary_logloss: 0.124918	valid_1's auc: 0.835689	valid_1's binary_logloss: 0.141524
    [18]	valid_0's auc: 0.878211	valid_0's binary_logloss: 0.124197	valid_1's auc: 0.835893	valid_1's binary_logloss: 0.141285
    [19]	valid_0's auc: 0.879125	valid_0's binary_logloss: 0.123553	valid_1's auc: 0.835877	valid_1's binary_logloss: 0.141128
    [20]	valid_0's auc: 0.880489	valid_0's binary_logloss: 0.122856	valid_1's auc: 0.835385	valid_1's binary_logloss: 0.141032
    [21]	valid_0's auc: 0.881696	valid_0's binary_logloss: 0.122219	valid_1's auc: 0.835822	valid_1's binary_logloss: 0.140843
    [22]	valid_0's auc: 0.882257	valid_0's binary_logloss: 0.121726	valid_1's auc: 0.835849	valid_1's binary_logloss: 0.140761
    [23]	valid_0's auc: 0.883635	valid_0's binary_logloss: 0.121206	valid_1's auc: 0.836223	valid_1's binary_logloss: 0.140607
    [24]	valid_0's auc: 0.884533	valid_0's binary_logloss: 0.120734	valid_1's auc: 0.836473	valid_1's binary_logloss: 0.14049
    [25]	valid_0's auc: 0.885234	valid_0's binary_logloss: 0.120268	valid_1's auc: 0.836722	valid_1's binary_logloss: 0.140403
    [26]	valid_0's auc: 0.886292	valid_0's binary_logloss: 0.119794	valid_1's auc: 0.836549	valid_1's binary_logloss: 0.140423
    [27]	valid_0's auc: 0.887064	valid_0's binary_logloss: 0.119366	valid_1's auc: 0.836155	valid_1's binary_logloss: 0.140447
    [28]	valid_0's auc: 0.887621	valid_0's binary_logloss: 0.119008	valid_1's auc: 0.835594	valid_1's binary_logloss: 0.140532
    [29]	valid_0's auc: 0.888965	valid_0's binary_logloss: 0.118547	valid_1's auc: 0.835464	valid_1's binary_logloss: 0.140508
    [30]	valid_0's auc: 0.889898	valid_0's binary_logloss: 0.118139	valid_1's auc: 0.83577	valid_1's binary_logloss: 0.140461
    [31]	valid_0's auc: 0.890896	valid_0's binary_logloss: 0.117734	valid_1's auc: 0.835475	valid_1's binary_logloss: 0.140463
    [32]	valid_0's auc: 0.892374	valid_0's binary_logloss: 0.1173	valid_1's auc: 0.835364	valid_1's binary_logloss: 0.140506
    [33]	valid_0's auc: 0.893164	valid_0's binary_logloss: 0.116978	valid_1's auc: 0.835865	valid_1's binary_logloss: 0.14041
    [34]	valid_0's auc: 0.893848	valid_0's binary_logloss: 0.11662	valid_1's auc: 0.836021	valid_1's binary_logloss: 0.140353
    [35]	valid_0's auc: 0.894232	valid_0's binary_logloss: 0.116323	valid_1's auc: 0.8359	valid_1's binary_logloss: 0.140396
    [36]	valid_0's auc: 0.895003	valid_0's binary_logloss: 0.115986	valid_1's auc: 0.835855	valid_1's binary_logloss: 0.140416
    [37]	valid_0's auc: 0.895898	valid_0's binary_logloss: 0.115609	valid_1's auc: 0.836185	valid_1's binary_logloss: 0.140369
    [38]	valid_0's auc: 0.896459	valid_0's binary_logloss: 0.11527	valid_1's auc: 0.835754	valid_1's binary_logloss: 0.140443
    [39]	valid_0's auc: 0.897377	valid_0's binary_logloss: 0.114873	valid_1's auc: 0.835638	valid_1's binary_logloss: 0.140474
    [40]	valid_0's auc: 0.89776	valid_0's binary_logloss: 0.114588	valid_1's auc: 0.835639	valid_1's binary_logloss: 0.140491
    [41]	valid_0's auc: 0.898583	valid_0's binary_logloss: 0.114302	valid_1's auc: 0.835705	valid_1's binary_logloss: 0.140506
    [42]	valid_0's auc: 0.899197	valid_0's binary_logloss: 0.113975	valid_1's auc: 0.835052	valid_1's binary_logloss: 0.14064
    [43]	valid_0's auc: 0.899803	valid_0's binary_logloss: 0.113654	valid_1's auc: 0.835035	valid_1's binary_logloss: 0.140691
    [44]	valid_0's auc: 0.900641	valid_0's binary_logloss: 0.113388	valid_1's auc: 0.835214	valid_1's binary_logloss: 0.140703
    [45]	valid_0's auc: 0.900962	valid_0's binary_logloss: 0.113098	valid_1's auc: 0.835276	valid_1's binary_logloss: 0.140695
    [46]	valid_0's auc: 0.901584	valid_0's binary_logloss: 0.112771	valid_1's auc: 0.83495	valid_1's binary_logloss: 0.140754
    [47]	valid_0's auc: 0.902256	valid_0's binary_logloss: 0.112493	valid_1's auc: 0.835639	valid_1's binary_logloss: 0.14064
    [48]	valid_0's auc: 0.902688	valid_0's binary_logloss: 0.112198	valid_1's auc: 0.835495	valid_1's binary_logloss: 0.140691
    [49]	valid_0's auc: 0.902922	valid_0's binary_logloss: 0.111944	valid_1's auc: 0.835281	valid_1's binary_logloss: 0.140819
    [50]	valid_0's auc: 0.903747	valid_0's binary_logloss: 0.111595	valid_1's auc: 0.835359	valid_1's binary_logloss: 0.140811
    [51]	valid_0's auc: 0.904427	valid_0's binary_logloss: 0.111354	valid_1's auc: 0.835245	valid_1's binary_logloss: 0.140873
    [52]	valid_0's auc: 0.90467	valid_0's binary_logloss: 0.111111	valid_1's auc: 0.835057	valid_1's binary_logloss: 0.140993
    [53]	valid_0's auc: 0.904868	valid_0's binary_logloss: 0.110853	valid_1's auc: 0.834751	valid_1's binary_logloss: 0.14108
    [54]	valid_0's auc: 0.905166	valid_0's binary_logloss: 0.110627	valid_1's auc: 0.83411	valid_1's binary_logloss: 0.141282
    [55]	valid_0's auc: 0.905665	valid_0's binary_logloss: 0.110375	valid_1's auc: 0.833739	valid_1's binary_logloss: 0.141413
    Early stopping, best iteration is:
    [25]	valid_0's auc: 0.885234	valid_0's binary_logloss: 0.120268	valid_1's auc: 0.836722	valid_1's binary_logloss: 0.140403
    [1]	valid_0's auc: 0.824873	valid_0's binary_logloss: 0.156222	valid_1's auc: 0.817791	valid_1's binary_logloss: 0.165072
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.828725	valid_0's binary_logloss: 0.151244	valid_1's auc: 0.822586	valid_1's binary_logloss: 0.160253
    [3]	valid_0's auc: 0.83594	valid_0's binary_logloss: 0.147423	valid_1's auc: 0.828474	valid_1's binary_logloss: 0.156542
    [4]	valid_0's auc: 0.839489	valid_0's binary_logloss: 0.144426	valid_1's auc: 0.831396	valid_1's binary_logloss: 0.153706
    [5]	valid_0's auc: 0.843358	valid_0's binary_logloss: 0.142067	valid_1's auc: 0.833466	valid_1's binary_logloss: 0.151399
    [6]	valid_0's auc: 0.845601	valid_0's binary_logloss: 0.14009	valid_1's auc: 0.833857	valid_1's binary_logloss: 0.149488
    [7]	valid_0's auc: 0.846477	valid_0's binary_logloss: 0.138491	valid_1's auc: 0.833143	valid_1's binary_logloss: 0.148023
    [8]	valid_0's auc: 0.847725	valid_0's binary_logloss: 0.137129	valid_1's auc: 0.833971	valid_1's binary_logloss: 0.146757
    [9]	valid_0's auc: 0.848442	valid_0's binary_logloss: 0.135908	valid_1's auc: 0.835976	valid_1's binary_logloss: 0.145685
    [10]	valid_0's auc: 0.849759	valid_0's binary_logloss: 0.134781	valid_1's auc: 0.836214	valid_1's binary_logloss: 0.144769
    [11]	valid_0's auc: 0.852238	valid_0's binary_logloss: 0.133835	valid_1's auc: 0.837243	valid_1's binary_logloss: 0.143925
    [12]	valid_0's auc: 0.853743	valid_0's binary_logloss: 0.132972	valid_1's auc: 0.836647	valid_1's binary_logloss: 0.143391
    [13]	valid_0's auc: 0.854568	valid_0's binary_logloss: 0.132256	valid_1's auc: 0.837182	valid_1's binary_logloss: 0.142849
    [14]	valid_0's auc: 0.855928	valid_0's binary_logloss: 0.131554	valid_1's auc: 0.835941	valid_1's binary_logloss: 0.142474
    [15]	valid_0's auc: 0.85712	valid_0's binary_logloss: 0.130984	valid_1's auc: 0.834938	valid_1's binary_logloss: 0.142198
    [16]	valid_0's auc: 0.858721	valid_0's binary_logloss: 0.130371	valid_1's auc: 0.83561	valid_1's binary_logloss: 0.141802
    [17]	valid_0's auc: 0.859281	valid_0's binary_logloss: 0.129877	valid_1's auc: 0.835146	valid_1's binary_logloss: 0.141605
    [18]	valid_0's auc: 0.859881	valid_0's binary_logloss: 0.129417	valid_1's auc: 0.835386	valid_1's binary_logloss: 0.14132
    [19]	valid_0's auc: 0.861409	valid_0's binary_logloss: 0.128929	valid_1's auc: 0.834974	valid_1's binary_logloss: 0.141151
    [20]	valid_0's auc: 0.862574	valid_0's binary_logloss: 0.128458	valid_1's auc: 0.834949	valid_1's binary_logloss: 0.140968
    [21]	valid_0's auc: 0.863262	valid_0's binary_logloss: 0.128069	valid_1's auc: 0.834616	valid_1's binary_logloss: 0.14086
    [22]	valid_0's auc: 0.864655	valid_0's binary_logloss: 0.127684	valid_1's auc: 0.834363	valid_1's binary_logloss: 0.140766
    [23]	valid_0's auc: 0.865247	valid_0's binary_logloss: 0.127349	valid_1's auc: 0.834317	valid_1's binary_logloss: 0.140688
    [24]	valid_0's auc: 0.865882	valid_0's binary_logloss: 0.12704	valid_1's auc: 0.833543	valid_1's binary_logloss: 0.14068
    [25]	valid_0's auc: 0.867496	valid_0's binary_logloss: 0.126629	valid_1's auc: 0.834195	valid_1's binary_logloss: 0.140539
    [26]	valid_0's auc: 0.867923	valid_0's binary_logloss: 0.126353	valid_1's auc: 0.834028	valid_1's binary_logloss: 0.140506
    [27]	valid_0's auc: 0.868685	valid_0's binary_logloss: 0.126058	valid_1's auc: 0.834718	valid_1's binary_logloss: 0.140359
    [28]	valid_0's auc: 0.869304	valid_0's binary_logloss: 0.125764	valid_1's auc: 0.834935	valid_1's binary_logloss: 0.140287
    [29]	valid_0's auc: 0.870037	valid_0's binary_logloss: 0.125514	valid_1's auc: 0.834481	valid_1's binary_logloss: 0.140258
    [30]	valid_0's auc: 0.870785	valid_0's binary_logloss: 0.125254	valid_1's auc: 0.834179	valid_1's binary_logloss: 0.140275
    [31]	valid_0's auc: 0.871706	valid_0's binary_logloss: 0.124992	valid_1's auc: 0.834475	valid_1's binary_logloss: 0.140205
    [32]	valid_0's auc: 0.872582	valid_0's binary_logloss: 0.124728	valid_1's auc: 0.834353	valid_1's binary_logloss: 0.140189
    [33]	valid_0's auc: 0.873445	valid_0's binary_logloss: 0.124481	valid_1's auc: 0.834592	valid_1's binary_logloss: 0.140082
    [34]	valid_0's auc: 0.874095	valid_0's binary_logloss: 0.12426	valid_1's auc: 0.83436	valid_1's binary_logloss: 0.140101
    [35]	valid_0's auc: 0.874869	valid_0's binary_logloss: 0.123982	valid_1's auc: 0.834045	valid_1's binary_logloss: 0.140151
    [36]	valid_0's auc: 0.875446	valid_0's binary_logloss: 0.123753	valid_1's auc: 0.834073	valid_1's binary_logloss: 0.140125
    [37]	valid_0's auc: 0.875763	valid_0's binary_logloss: 0.123587	valid_1's auc: 0.833611	valid_1's binary_logloss: 0.140201
    [38]	valid_0's auc: 0.876603	valid_0's binary_logloss: 0.123335	valid_1's auc: 0.833805	valid_1's binary_logloss: 0.140159
    [39]	valid_0's auc: 0.877126	valid_0's binary_logloss: 0.123134	valid_1's auc: 0.834422	valid_1's binary_logloss: 0.140048
    [40]	valid_0's auc: 0.877575	valid_0's binary_logloss: 0.123013	valid_1's auc: 0.834343	valid_1's binary_logloss: 0.140069
    [41]	valid_0's auc: 0.87809	valid_0's binary_logloss: 0.122813	valid_1's auc: 0.834199	valid_1's binary_logloss: 0.140085
    Early stopping, best iteration is:
    [11]	valid_0's auc: 0.852238	valid_0's binary_logloss: 0.133835	valid_1's auc: 0.837243	valid_1's binary_logloss: 0.143925
    [1]	valid_0's auc: 0.821831	valid_0's binary_logloss: 0.156466	valid_1's auc: 0.817525	valid_1's binary_logloss: 0.165186
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.831974	valid_0's binary_logloss: 0.151137	valid_1's auc: 0.82532	valid_1's binary_logloss: 0.159691
    [3]	valid_0's auc: 0.839496	valid_0's binary_logloss: 0.14733	valid_1's auc: 0.831946	valid_1's binary_logloss: 0.156
    [4]	valid_0's auc: 0.843984	valid_0's binary_logloss: 0.144371	valid_1's auc: 0.834064	valid_1's binary_logloss: 0.153082
    [5]	valid_0's auc: 0.845854	valid_0's binary_logloss: 0.142024	valid_1's auc: 0.836918	valid_1's binary_logloss: 0.150735
    [6]	valid_0's auc: 0.848041	valid_0's binary_logloss: 0.140009	valid_1's auc: 0.838831	valid_1's binary_logloss: 0.148771
    [7]	valid_0's auc: 0.849655	valid_0's binary_logloss: 0.138307	valid_1's auc: 0.839111	valid_1's binary_logloss: 0.147373
    [8]	valid_0's auc: 0.85185	valid_0's binary_logloss: 0.136891	valid_1's auc: 0.838955	valid_1's binary_logloss: 0.146094
    [9]	valid_0's auc: 0.853067	valid_0's binary_logloss: 0.135655	valid_1's auc: 0.838081	valid_1's binary_logloss: 0.14516
    [10]	valid_0's auc: 0.853922	valid_0's binary_logloss: 0.134622	valid_1's auc: 0.837333	valid_1's binary_logloss: 0.144318
    [11]	valid_0's auc: 0.854729	valid_0's binary_logloss: 0.133702	valid_1's auc: 0.83725	valid_1's binary_logloss: 0.143512
    [12]	valid_0's auc: 0.856303	valid_0's binary_logloss: 0.132789	valid_1's auc: 0.837602	valid_1's binary_logloss: 0.142833
    [13]	valid_0's auc: 0.857206	valid_0's binary_logloss: 0.132038	valid_1's auc: 0.837364	valid_1's binary_logloss: 0.142245
    [14]	valid_0's auc: 0.858161	valid_0's binary_logloss: 0.131391	valid_1's auc: 0.83777	valid_1's binary_logloss: 0.141759
    [15]	valid_0's auc: 0.858975	valid_0's binary_logloss: 0.130772	valid_1's auc: 0.837831	valid_1's binary_logloss: 0.14139
    [16]	valid_0's auc: 0.859623	valid_0's binary_logloss: 0.130219	valid_1's auc: 0.837953	valid_1's binary_logloss: 0.141016
    [17]	valid_0's auc: 0.860576	valid_0's binary_logloss: 0.129684	valid_1's auc: 0.837985	valid_1's binary_logloss: 0.140713
    [18]	valid_0's auc: 0.861311	valid_0's binary_logloss: 0.129202	valid_1's auc: 0.83796	valid_1's binary_logloss: 0.140452
    [19]	valid_0's auc: 0.862347	valid_0's binary_logloss: 0.128715	valid_1's auc: 0.838506	valid_1's binary_logloss: 0.140189
    [20]	valid_0's auc: 0.86305	valid_0's binary_logloss: 0.128312	valid_1's auc: 0.837702	valid_1's binary_logloss: 0.140094
    [21]	valid_0's auc: 0.863758	valid_0's binary_logloss: 0.127907	valid_1's auc: 0.838127	valid_1's binary_logloss: 0.139858
    [22]	valid_0's auc: 0.864635	valid_0's binary_logloss: 0.127525	valid_1's auc: 0.838331	valid_1's binary_logloss: 0.139696
    [23]	valid_0's auc: 0.865866	valid_0's binary_logloss: 0.127143	valid_1's auc: 0.837841	valid_1's binary_logloss: 0.139625
    [24]	valid_0's auc: 0.867054	valid_0's binary_logloss: 0.126749	valid_1's auc: 0.838187	valid_1's binary_logloss: 0.139526
    [25]	valid_0's auc: 0.867553	valid_0's binary_logloss: 0.126476	valid_1's auc: 0.838308	valid_1's binary_logloss: 0.13949
    [26]	valid_0's auc: 0.868108	valid_0's binary_logloss: 0.126164	valid_1's auc: 0.838035	valid_1's binary_logloss: 0.139426
    [27]	valid_0's auc: 0.869014	valid_0's binary_logloss: 0.125868	valid_1's auc: 0.837545	valid_1's binary_logloss: 0.139445
    [28]	valid_0's auc: 0.869797	valid_0's binary_logloss: 0.12559	valid_1's auc: 0.837894	valid_1's binary_logloss: 0.139419
    [29]	valid_0's auc: 0.870435	valid_0's binary_logloss: 0.1253	valid_1's auc: 0.838103	valid_1's binary_logloss: 0.139321
    [30]	valid_0's auc: 0.87141	valid_0's binary_logloss: 0.125025	valid_1's auc: 0.838164	valid_1's binary_logloss: 0.139275
    [31]	valid_0's auc: 0.872143	valid_0's binary_logloss: 0.124769	valid_1's auc: 0.837843	valid_1's binary_logloss: 0.139285
    [32]	valid_0's auc: 0.872606	valid_0's binary_logloss: 0.124561	valid_1's auc: 0.837662	valid_1's binary_logloss: 0.139274
    [33]	valid_0's auc: 0.873337	valid_0's binary_logloss: 0.124346	valid_1's auc: 0.837661	valid_1's binary_logloss: 0.139284
    [34]	valid_0's auc: 0.873965	valid_0's binary_logloss: 0.124108	valid_1's auc: 0.837639	valid_1's binary_logloss: 0.139263
    [35]	valid_0's auc: 0.87457	valid_0's binary_logloss: 0.123857	valid_1's auc: 0.838159	valid_1's binary_logloss: 0.139137
    [36]	valid_0's auc: 0.874973	valid_0's binary_logloss: 0.123651	valid_1's auc: 0.838114	valid_1's binary_logloss: 0.139148
    [37]	valid_0's auc: 0.875657	valid_0's binary_logloss: 0.123447	valid_1's auc: 0.838519	valid_1's binary_logloss: 0.139109
    Early stopping, best iteration is:
    [7]	valid_0's auc: 0.849655	valid_0's binary_logloss: 0.138307	valid_1's auc: 0.839111	valid_1's binary_logloss: 0.147373
    [1]	valid_0's auc: 0.821427	valid_0's binary_logloss: 0.156592	valid_1's auc: 0.81711	valid_1's binary_logloss: 0.165273
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.827893	valid_0's binary_logloss: 0.151336	valid_1's auc: 0.820533	valid_1's binary_logloss: 0.160243
    [3]	valid_0's auc: 0.83753	valid_0's binary_logloss: 0.147487	valid_1's auc: 0.82841	valid_1's binary_logloss: 0.156547
    [4]	valid_0's auc: 0.84038	valid_0's binary_logloss: 0.144428	valid_1's auc: 0.8313	valid_1's binary_logloss: 0.153575
    [5]	valid_0's auc: 0.842945	valid_0's binary_logloss: 0.142089	valid_1's auc: 0.833579	valid_1's binary_logloss: 0.151354
    [6]	valid_0's auc: 0.843246	valid_0's binary_logloss: 0.140186	valid_1's auc: 0.833781	valid_1's binary_logloss: 0.14953
    [7]	valid_0's auc: 0.844301	valid_0's binary_logloss: 0.138471	valid_1's auc: 0.834317	valid_1's binary_logloss: 0.147954
    [8]	valid_0's auc: 0.846945	valid_0's binary_logloss: 0.137078	valid_1's auc: 0.834895	valid_1's binary_logloss: 0.146786
    [9]	valid_0's auc: 0.849381	valid_0's binary_logloss: 0.135906	valid_1's auc: 0.834922	valid_1's binary_logloss: 0.145762
    [10]	valid_0's auc: 0.850944	valid_0's binary_logloss: 0.134855	valid_1's auc: 0.835441	valid_1's binary_logloss: 0.144958
    [11]	valid_0's auc: 0.852557	valid_0's binary_logloss: 0.133895	valid_1's auc: 0.835103	valid_1's binary_logloss: 0.144293
    [12]	valid_0's auc: 0.854609	valid_0's binary_logloss: 0.133013	valid_1's auc: 0.835686	valid_1's binary_logloss: 0.143793
    [13]	valid_0's auc: 0.855817	valid_0's binary_logloss: 0.132247	valid_1's auc: 0.835296	valid_1's binary_logloss: 0.143302
    [14]	valid_0's auc: 0.857501	valid_0's binary_logloss: 0.131545	valid_1's auc: 0.836432	valid_1's binary_logloss: 0.142761
    [15]	valid_0's auc: 0.858907	valid_0's binary_logloss: 0.130878	valid_1's auc: 0.836329	valid_1's binary_logloss: 0.142383
    [16]	valid_0's auc: 0.859887	valid_0's binary_logloss: 0.130287	valid_1's auc: 0.836611	valid_1's binary_logloss: 0.141883
    [17]	valid_0's auc: 0.860889	valid_0's binary_logloss: 0.129757	valid_1's auc: 0.836848	valid_1's binary_logloss: 0.141535
    [18]	valid_0's auc: 0.861827	valid_0's binary_logloss: 0.129301	valid_1's auc: 0.837106	valid_1's binary_logloss: 0.141257
    [19]	valid_0's auc: 0.862972	valid_0's binary_logloss: 0.128826	valid_1's auc: 0.837185	valid_1's binary_logloss: 0.141043
    [20]	valid_0's auc: 0.864083	valid_0's binary_logloss: 0.128369	valid_1's auc: 0.837509	valid_1's binary_logloss: 0.140794
    [21]	valid_0's auc: 0.864747	valid_0's binary_logloss: 0.127959	valid_1's auc: 0.837888	valid_1's binary_logloss: 0.140626
    [22]	valid_0's auc: 0.865769	valid_0's binary_logloss: 0.127562	valid_1's auc: 0.837811	valid_1's binary_logloss: 0.140487
    [23]	valid_0's auc: 0.866657	valid_0's binary_logloss: 0.127217	valid_1's auc: 0.837884	valid_1's binary_logloss: 0.140328
    [24]	valid_0's auc: 0.867293	valid_0's binary_logloss: 0.126875	valid_1's auc: 0.838481	valid_1's binary_logloss: 0.140215
    [25]	valid_0's auc: 0.867983	valid_0's binary_logloss: 0.126562	valid_1's auc: 0.838239	valid_1's binary_logloss: 0.140124
    [26]	valid_0's auc: 0.868559	valid_0's binary_logloss: 0.126248	valid_1's auc: 0.837903	valid_1's binary_logloss: 0.140092
    [27]	valid_0's auc: 0.869394	valid_0's binary_logloss: 0.125936	valid_1's auc: 0.837493	valid_1's binary_logloss: 0.14006
    [28]	valid_0's auc: 0.87048	valid_0's binary_logloss: 0.125677	valid_1's auc: 0.837623	valid_1's binary_logloss: 0.140007
    [29]	valid_0's auc: 0.87105	valid_0's binary_logloss: 0.125405	valid_1's auc: 0.838216	valid_1's binary_logloss: 0.13986
    [30]	valid_0's auc: 0.871749	valid_0's binary_logloss: 0.125147	valid_1's auc: 0.838898	valid_1's binary_logloss: 0.139742
    [31]	valid_0's auc: 0.87247	valid_0's binary_logloss: 0.124907	valid_1's auc: 0.838959	valid_1's binary_logloss: 0.139727
    [32]	valid_0's auc: 0.87282	valid_0's binary_logloss: 0.124724	valid_1's auc: 0.838675	valid_1's binary_logloss: 0.139761
    [33]	valid_0's auc: 0.874106	valid_0's binary_logloss: 0.124412	valid_1's auc: 0.838893	valid_1's binary_logloss: 0.139687
    [34]	valid_0's auc: 0.874887	valid_0's binary_logloss: 0.124169	valid_1's auc: 0.838801	valid_1's binary_logloss: 0.139672
    [35]	valid_0's auc: 0.875447	valid_0's binary_logloss: 0.123934	valid_1's auc: 0.838835	valid_1's binary_logloss: 0.139667
    [36]	valid_0's auc: 0.87617	valid_0's binary_logloss: 0.123693	valid_1's auc: 0.838505	valid_1's binary_logloss: 0.139699
    [37]	valid_0's auc: 0.876793	valid_0's binary_logloss: 0.12346	valid_1's auc: 0.838104	valid_1's binary_logloss: 0.139783
    [38]	valid_0's auc: 0.877265	valid_0's binary_logloss: 0.123251	valid_1's auc: 0.838267	valid_1's binary_logloss: 0.139787
    [39]	valid_0's auc: 0.877869	valid_0's binary_logloss: 0.123018	valid_1's auc: 0.838004	valid_1's binary_logloss: 0.139806
    [40]	valid_0's auc: 0.878509	valid_0's binary_logloss: 0.122803	valid_1's auc: 0.838086	valid_1's binary_logloss: 0.139745
    [41]	valid_0's auc: 0.879077	valid_0's binary_logloss: 0.122585	valid_1's auc: 0.838538	valid_1's binary_logloss: 0.139694
    [42]	valid_0's auc: 0.879515	valid_0's binary_logloss: 0.122368	valid_1's auc: 0.838647	valid_1's binary_logloss: 0.139655
    [43]	valid_0's auc: 0.879985	valid_0's binary_logloss: 0.122166	valid_1's auc: 0.838495	valid_1's binary_logloss: 0.139653
    [44]	valid_0's auc: 0.88041	valid_0's binary_logloss: 0.121985	valid_1's auc: 0.838221	valid_1's binary_logloss: 0.139755
    [45]	valid_0's auc: 0.880907	valid_0's binary_logloss: 0.121777	valid_1's auc: 0.837981	valid_1's binary_logloss: 0.139769
    [46]	valid_0's auc: 0.881216	valid_0's binary_logloss: 0.121594	valid_1's auc: 0.838471	valid_1's binary_logloss: 0.139693
    [47]	valid_0's auc: 0.881591	valid_0's binary_logloss: 0.121422	valid_1's auc: 0.83861	valid_1's binary_logloss: 0.139687
    [48]	valid_0's auc: 0.881867	valid_0's binary_logloss: 0.121266	valid_1's auc: 0.838593	valid_1's binary_logloss: 0.139682
    [49]	valid_0's auc: 0.882285	valid_0's binary_logloss: 0.121041	valid_1's auc: 0.838317	valid_1's binary_logloss: 0.139741
    [50]	valid_0's auc: 0.882828	valid_0's binary_logloss: 0.120853	valid_1's auc: 0.838244	valid_1's binary_logloss: 0.139759
    [51]	valid_0's auc: 0.883154	valid_0's binary_logloss: 0.120688	valid_1's auc: 0.838222	valid_1's binary_logloss: 0.139803
    [52]	valid_0's auc: 0.883348	valid_0's binary_logloss: 0.120567	valid_1's auc: 0.838064	valid_1's binary_logloss: 0.139824
    [53]	valid_0's auc: 0.883583	valid_0's binary_logloss: 0.120424	valid_1's auc: 0.83788	valid_1's binary_logloss: 0.139844
    [54]	valid_0's auc: 0.884106	valid_0's binary_logloss: 0.120208	valid_1's auc: 0.837625	valid_1's binary_logloss: 0.139886
    [55]	valid_0's auc: 0.884777	valid_0's binary_logloss: 0.120039	valid_1's auc: 0.837585	valid_1's binary_logloss: 0.139902
    [56]	valid_0's auc: 0.88511	valid_0's binary_logloss: 0.11989	valid_1's auc: 0.837646	valid_1's binary_logloss: 0.139926
    [57]	valid_0's auc: 0.885365	valid_0's binary_logloss: 0.11975	valid_1's auc: 0.837639	valid_1's binary_logloss: 0.139934
    [58]	valid_0's auc: 0.885606	valid_0's binary_logloss: 0.119595	valid_1's auc: 0.837726	valid_1's binary_logloss: 0.139938
    [59]	valid_0's auc: 0.885965	valid_0's binary_logloss: 0.119403	valid_1's auc: 0.837558	valid_1's binary_logloss: 0.140007
    [60]	valid_0's auc: 0.886208	valid_0's binary_logloss: 0.119263	valid_1's auc: 0.83744	valid_1's binary_logloss: 0.140079
    [61]	valid_0's auc: 0.886458	valid_0's binary_logloss: 0.119118	valid_1's auc: 0.837349	valid_1's binary_logloss: 0.140059
    Early stopping, best iteration is:
    [31]	valid_0's auc: 0.87247	valid_0's binary_logloss: 0.124907	valid_1's auc: 0.838959	valid_1's binary_logloss: 0.139727
    [1]	valid_0's auc: 0.824873	valid_0's binary_logloss: 0.156222	valid_1's auc: 0.817791	valid_1's binary_logloss: 0.165072
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.828725	valid_0's binary_logloss: 0.151244	valid_1's auc: 0.822586	valid_1's binary_logloss: 0.160253
    [3]	valid_0's auc: 0.83594	valid_0's binary_logloss: 0.147423	valid_1's auc: 0.828474	valid_1's binary_logloss: 0.156542
    [4]	valid_0's auc: 0.839489	valid_0's binary_logloss: 0.144426	valid_1's auc: 0.831396	valid_1's binary_logloss: 0.153706
    [5]	valid_0's auc: 0.843358	valid_0's binary_logloss: 0.142067	valid_1's auc: 0.833466	valid_1's binary_logloss: 0.151399
    [6]	valid_0's auc: 0.845601	valid_0's binary_logloss: 0.14009	valid_1's auc: 0.833857	valid_1's binary_logloss: 0.149488
    [7]	valid_0's auc: 0.846477	valid_0's binary_logloss: 0.138491	valid_1's auc: 0.833143	valid_1's binary_logloss: 0.148023
    [8]	valid_0's auc: 0.847725	valid_0's binary_logloss: 0.137129	valid_1's auc: 0.833971	valid_1's binary_logloss: 0.146757
    [9]	valid_0's auc: 0.848442	valid_0's binary_logloss: 0.135908	valid_1's auc: 0.835976	valid_1's binary_logloss: 0.145685
    [10]	valid_0's auc: 0.849759	valid_0's binary_logloss: 0.134781	valid_1's auc: 0.836214	valid_1's binary_logloss: 0.144769
    [11]	valid_0's auc: 0.852238	valid_0's binary_logloss: 0.133835	valid_1's auc: 0.837243	valid_1's binary_logloss: 0.143925
    [12]	valid_0's auc: 0.853743	valid_0's binary_logloss: 0.132972	valid_1's auc: 0.836647	valid_1's binary_logloss: 0.143391
    [13]	valid_0's auc: 0.854568	valid_0's binary_logloss: 0.132256	valid_1's auc: 0.837182	valid_1's binary_logloss: 0.142849
    [14]	valid_0's auc: 0.855928	valid_0's binary_logloss: 0.131554	valid_1's auc: 0.835941	valid_1's binary_logloss: 0.142474
    [15]	valid_0's auc: 0.85712	valid_0's binary_logloss: 0.130984	valid_1's auc: 0.834938	valid_1's binary_logloss: 0.142198
    [16]	valid_0's auc: 0.858721	valid_0's binary_logloss: 0.130371	valid_1's auc: 0.83561	valid_1's binary_logloss: 0.141802
    [17]	valid_0's auc: 0.859281	valid_0's binary_logloss: 0.129877	valid_1's auc: 0.835146	valid_1's binary_logloss: 0.141605
    [18]	valid_0's auc: 0.859881	valid_0's binary_logloss: 0.129417	valid_1's auc: 0.835386	valid_1's binary_logloss: 0.14132
    [19]	valid_0's auc: 0.861409	valid_0's binary_logloss: 0.128929	valid_1's auc: 0.834974	valid_1's binary_logloss: 0.141151
    [20]	valid_0's auc: 0.862574	valid_0's binary_logloss: 0.128458	valid_1's auc: 0.834949	valid_1's binary_logloss: 0.140968
    [21]	valid_0's auc: 0.863262	valid_0's binary_logloss: 0.128069	valid_1's auc: 0.834616	valid_1's binary_logloss: 0.14086
    [22]	valid_0's auc: 0.864655	valid_0's binary_logloss: 0.127684	valid_1's auc: 0.834363	valid_1's binary_logloss: 0.140766
    [23]	valid_0's auc: 0.865247	valid_0's binary_logloss: 0.127349	valid_1's auc: 0.834317	valid_1's binary_logloss: 0.140688
    [24]	valid_0's auc: 0.865882	valid_0's binary_logloss: 0.12704	valid_1's auc: 0.833543	valid_1's binary_logloss: 0.14068
    [25]	valid_0's auc: 0.867496	valid_0's binary_logloss: 0.126629	valid_1's auc: 0.834195	valid_1's binary_logloss: 0.140539
    [26]	valid_0's auc: 0.867923	valid_0's binary_logloss: 0.126353	valid_1's auc: 0.834028	valid_1's binary_logloss: 0.140506
    [27]	valid_0's auc: 0.868685	valid_0's binary_logloss: 0.126058	valid_1's auc: 0.834718	valid_1's binary_logloss: 0.140359
    [28]	valid_0's auc: 0.869304	valid_0's binary_logloss: 0.125764	valid_1's auc: 0.834935	valid_1's binary_logloss: 0.140287
    [29]	valid_0's auc: 0.870037	valid_0's binary_logloss: 0.125514	valid_1's auc: 0.834481	valid_1's binary_logloss: 0.140258
    [30]	valid_0's auc: 0.870785	valid_0's binary_logloss: 0.125254	valid_1's auc: 0.834179	valid_1's binary_logloss: 0.140275
    [31]	valid_0's auc: 0.871706	valid_0's binary_logloss: 0.124992	valid_1's auc: 0.834475	valid_1's binary_logloss: 0.140205
    [32]	valid_0's auc: 0.872582	valid_0's binary_logloss: 0.124728	valid_1's auc: 0.834353	valid_1's binary_logloss: 0.140189
    [33]	valid_0's auc: 0.873445	valid_0's binary_logloss: 0.124481	valid_1's auc: 0.834592	valid_1's binary_logloss: 0.140082
    [34]	valid_0's auc: 0.874095	valid_0's binary_logloss: 0.12426	valid_1's auc: 0.83436	valid_1's binary_logloss: 0.140101
    [35]	valid_0's auc: 0.874869	valid_0's binary_logloss: 0.123982	valid_1's auc: 0.834045	valid_1's binary_logloss: 0.140151
    [36]	valid_0's auc: 0.875446	valid_0's binary_logloss: 0.123753	valid_1's auc: 0.834073	valid_1's binary_logloss: 0.140125
    [37]	valid_0's auc: 0.875763	valid_0's binary_logloss: 0.123587	valid_1's auc: 0.833611	valid_1's binary_logloss: 0.140201
    [38]	valid_0's auc: 0.876603	valid_0's binary_logloss: 0.123335	valid_1's auc: 0.833805	valid_1's binary_logloss: 0.140159
    [39]	valid_0's auc: 0.877126	valid_0's binary_logloss: 0.123134	valid_1's auc: 0.834422	valid_1's binary_logloss: 0.140048
    [40]	valid_0's auc: 0.877575	valid_0's binary_logloss: 0.123013	valid_1's auc: 0.834343	valid_1's binary_logloss: 0.140069
    [41]	valid_0's auc: 0.87809	valid_0's binary_logloss: 0.122813	valid_1's auc: 0.834199	valid_1's binary_logloss: 0.140085
    Early stopping, best iteration is:
    [11]	valid_0's auc: 0.852238	valid_0's binary_logloss: 0.133835	valid_1's auc: 0.837243	valid_1's binary_logloss: 0.143925
    [1]	valid_0's auc: 0.821831	valid_0's binary_logloss: 0.156466	valid_1's auc: 0.817525	valid_1's binary_logloss: 0.165186
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.831974	valid_0's binary_logloss: 0.151137	valid_1's auc: 0.82532	valid_1's binary_logloss: 0.159691
    [3]	valid_0's auc: 0.839496	valid_0's binary_logloss: 0.14733	valid_1's auc: 0.831946	valid_1's binary_logloss: 0.156
    [4]	valid_0's auc: 0.843984	valid_0's binary_logloss: 0.144371	valid_1's auc: 0.834064	valid_1's binary_logloss: 0.153082
    [5]	valid_0's auc: 0.845854	valid_0's binary_logloss: 0.142024	valid_1's auc: 0.836918	valid_1's binary_logloss: 0.150735
    [6]	valid_0's auc: 0.848041	valid_0's binary_logloss: 0.140009	valid_1's auc: 0.838831	valid_1's binary_logloss: 0.148771
    [7]	valid_0's auc: 0.849655	valid_0's binary_logloss: 0.138307	valid_1's auc: 0.839111	valid_1's binary_logloss: 0.147373
    [8]	valid_0's auc: 0.85185	valid_0's binary_logloss: 0.136891	valid_1's auc: 0.838955	valid_1's binary_logloss: 0.146094
    [9]	valid_0's auc: 0.853067	valid_0's binary_logloss: 0.135655	valid_1's auc: 0.838081	valid_1's binary_logloss: 0.14516
    [10]	valid_0's auc: 0.853922	valid_0's binary_logloss: 0.134622	valid_1's auc: 0.837333	valid_1's binary_logloss: 0.144318
    [11]	valid_0's auc: 0.854729	valid_0's binary_logloss: 0.133702	valid_1's auc: 0.83725	valid_1's binary_logloss: 0.143512
    [12]	valid_0's auc: 0.856303	valid_0's binary_logloss: 0.132789	valid_1's auc: 0.837602	valid_1's binary_logloss: 0.142833
    [13]	valid_0's auc: 0.857206	valid_0's binary_logloss: 0.132038	valid_1's auc: 0.837364	valid_1's binary_logloss: 0.142245
    [14]	valid_0's auc: 0.858161	valid_0's binary_logloss: 0.131391	valid_1's auc: 0.83777	valid_1's binary_logloss: 0.141759
    [15]	valid_0's auc: 0.858975	valid_0's binary_logloss: 0.130772	valid_1's auc: 0.837831	valid_1's binary_logloss: 0.14139
    [16]	valid_0's auc: 0.859623	valid_0's binary_logloss: 0.130219	valid_1's auc: 0.837953	valid_1's binary_logloss: 0.141016
    [17]	valid_0's auc: 0.860576	valid_0's binary_logloss: 0.129684	valid_1's auc: 0.837985	valid_1's binary_logloss: 0.140713
    [18]	valid_0's auc: 0.861311	valid_0's binary_logloss: 0.129202	valid_1's auc: 0.83796	valid_1's binary_logloss: 0.140452
    [19]	valid_0's auc: 0.862347	valid_0's binary_logloss: 0.128715	valid_1's auc: 0.838506	valid_1's binary_logloss: 0.140189
    [20]	valid_0's auc: 0.86305	valid_0's binary_logloss: 0.128312	valid_1's auc: 0.837702	valid_1's binary_logloss: 0.140094
    [21]	valid_0's auc: 0.863758	valid_0's binary_logloss: 0.127907	valid_1's auc: 0.838127	valid_1's binary_logloss: 0.139858
    [22]	valid_0's auc: 0.864635	valid_0's binary_logloss: 0.127525	valid_1's auc: 0.838331	valid_1's binary_logloss: 0.139696
    [23]	valid_0's auc: 0.865866	valid_0's binary_logloss: 0.127143	valid_1's auc: 0.837841	valid_1's binary_logloss: 0.139625
    [24]	valid_0's auc: 0.867054	valid_0's binary_logloss: 0.126749	valid_1's auc: 0.838187	valid_1's binary_logloss: 0.139526
    [25]	valid_0's auc: 0.867553	valid_0's binary_logloss: 0.126476	valid_1's auc: 0.838308	valid_1's binary_logloss: 0.13949
    [26]	valid_0's auc: 0.868108	valid_0's binary_logloss: 0.126164	valid_1's auc: 0.838035	valid_1's binary_logloss: 0.139426
    [27]	valid_0's auc: 0.869014	valid_0's binary_logloss: 0.125868	valid_1's auc: 0.837545	valid_1's binary_logloss: 0.139445
    [28]	valid_0's auc: 0.869797	valid_0's binary_logloss: 0.12559	valid_1's auc: 0.837894	valid_1's binary_logloss: 0.139419
    [29]	valid_0's auc: 0.870435	valid_0's binary_logloss: 0.1253	valid_1's auc: 0.838103	valid_1's binary_logloss: 0.139321
    [30]	valid_0's auc: 0.87141	valid_0's binary_logloss: 0.125025	valid_1's auc: 0.838164	valid_1's binary_logloss: 0.139275
    [31]	valid_0's auc: 0.872143	valid_0's binary_logloss: 0.124769	valid_1's auc: 0.837843	valid_1's binary_logloss: 0.139285
    [32]	valid_0's auc: 0.872606	valid_0's binary_logloss: 0.124561	valid_1's auc: 0.837662	valid_1's binary_logloss: 0.139274
    [33]	valid_0's auc: 0.873337	valid_0's binary_logloss: 0.124346	valid_1's auc: 0.837661	valid_1's binary_logloss: 0.139284
    [34]	valid_0's auc: 0.873965	valid_0's binary_logloss: 0.124108	valid_1's auc: 0.837639	valid_1's binary_logloss: 0.139263
    [35]	valid_0's auc: 0.87457	valid_0's binary_logloss: 0.123857	valid_1's auc: 0.838159	valid_1's binary_logloss: 0.139137
    [36]	valid_0's auc: 0.874973	valid_0's binary_logloss: 0.123651	valid_1's auc: 0.838114	valid_1's binary_logloss: 0.139148
    [37]	valid_0's auc: 0.875657	valid_0's binary_logloss: 0.123447	valid_1's auc: 0.838519	valid_1's binary_logloss: 0.139109
    Early stopping, best iteration is:
    [7]	valid_0's auc: 0.849655	valid_0's binary_logloss: 0.138307	valid_1's auc: 0.839111	valid_1's binary_logloss: 0.147373
    [1]	valid_0's auc: 0.821427	valid_0's binary_logloss: 0.156592	valid_1's auc: 0.81711	valid_1's binary_logloss: 0.165273
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.827893	valid_0's binary_logloss: 0.151336	valid_1's auc: 0.820533	valid_1's binary_logloss: 0.160243
    [3]	valid_0's auc: 0.83753	valid_0's binary_logloss: 0.147487	valid_1's auc: 0.82841	valid_1's binary_logloss: 0.156547
    [4]	valid_0's auc: 0.84038	valid_0's binary_logloss: 0.144428	valid_1's auc: 0.8313	valid_1's binary_logloss: 0.153575
    [5]	valid_0's auc: 0.842945	valid_0's binary_logloss: 0.142089	valid_1's auc: 0.833579	valid_1's binary_logloss: 0.151354
    [6]	valid_0's auc: 0.843246	valid_0's binary_logloss: 0.140186	valid_1's auc: 0.833781	valid_1's binary_logloss: 0.14953
    [7]	valid_0's auc: 0.844301	valid_0's binary_logloss: 0.138471	valid_1's auc: 0.834317	valid_1's binary_logloss: 0.147954
    [8]	valid_0's auc: 0.846945	valid_0's binary_logloss: 0.137078	valid_1's auc: 0.834895	valid_1's binary_logloss: 0.146786
    [9]	valid_0's auc: 0.849381	valid_0's binary_logloss: 0.135906	valid_1's auc: 0.834922	valid_1's binary_logloss: 0.145762
    [10]	valid_0's auc: 0.850944	valid_0's binary_logloss: 0.134855	valid_1's auc: 0.835441	valid_1's binary_logloss: 0.144958
    [11]	valid_0's auc: 0.852557	valid_0's binary_logloss: 0.133895	valid_1's auc: 0.835103	valid_1's binary_logloss: 0.144293
    [12]	valid_0's auc: 0.854609	valid_0's binary_logloss: 0.133013	valid_1's auc: 0.835686	valid_1's binary_logloss: 0.143793
    [13]	valid_0's auc: 0.855817	valid_0's binary_logloss: 0.132247	valid_1's auc: 0.835296	valid_1's binary_logloss: 0.143302
    [14]	valid_0's auc: 0.857501	valid_0's binary_logloss: 0.131545	valid_1's auc: 0.836432	valid_1's binary_logloss: 0.142761
    [15]	valid_0's auc: 0.858907	valid_0's binary_logloss: 0.130878	valid_1's auc: 0.836329	valid_1's binary_logloss: 0.142383
    [16]	valid_0's auc: 0.859887	valid_0's binary_logloss: 0.130287	valid_1's auc: 0.836611	valid_1's binary_logloss: 0.141883
    [17]	valid_0's auc: 0.860889	valid_0's binary_logloss: 0.129757	valid_1's auc: 0.836848	valid_1's binary_logloss: 0.141535
    [18]	valid_0's auc: 0.861827	valid_0's binary_logloss: 0.129301	valid_1's auc: 0.837106	valid_1's binary_logloss: 0.141257
    [19]	valid_0's auc: 0.862972	valid_0's binary_logloss: 0.128826	valid_1's auc: 0.837185	valid_1's binary_logloss: 0.141043
    [20]	valid_0's auc: 0.864083	valid_0's binary_logloss: 0.128369	valid_1's auc: 0.837509	valid_1's binary_logloss: 0.140794
    [21]	valid_0's auc: 0.864747	valid_0's binary_logloss: 0.127959	valid_1's auc: 0.837888	valid_1's binary_logloss: 0.140626
    [22]	valid_0's auc: 0.865769	valid_0's binary_logloss: 0.127562	valid_1's auc: 0.837811	valid_1's binary_logloss: 0.140487
    [23]	valid_0's auc: 0.866657	valid_0's binary_logloss: 0.127217	valid_1's auc: 0.837884	valid_1's binary_logloss: 0.140328
    [24]	valid_0's auc: 0.867293	valid_0's binary_logloss: 0.126875	valid_1's auc: 0.838481	valid_1's binary_logloss: 0.140215
    [25]	valid_0's auc: 0.867983	valid_0's binary_logloss: 0.126562	valid_1's auc: 0.838239	valid_1's binary_logloss: 0.140124
    [26]	valid_0's auc: 0.868559	valid_0's binary_logloss: 0.126248	valid_1's auc: 0.837903	valid_1's binary_logloss: 0.140092
    [27]	valid_0's auc: 0.869394	valid_0's binary_logloss: 0.125936	valid_1's auc: 0.837493	valid_1's binary_logloss: 0.14006
    [28]	valid_0's auc: 0.87048	valid_0's binary_logloss: 0.125677	valid_1's auc: 0.837623	valid_1's binary_logloss: 0.140007
    [29]	valid_0's auc: 0.87105	valid_0's binary_logloss: 0.125405	valid_1's auc: 0.838216	valid_1's binary_logloss: 0.13986
    [30]	valid_0's auc: 0.871749	valid_0's binary_logloss: 0.125147	valid_1's auc: 0.838898	valid_1's binary_logloss: 0.139742
    [31]	valid_0's auc: 0.87247	valid_0's binary_logloss: 0.124907	valid_1's auc: 0.838959	valid_1's binary_logloss: 0.139727
    [32]	valid_0's auc: 0.87282	valid_0's binary_logloss: 0.124724	valid_1's auc: 0.838675	valid_1's binary_logloss: 0.139761
    [33]	valid_0's auc: 0.874106	valid_0's binary_logloss: 0.124412	valid_1's auc: 0.838893	valid_1's binary_logloss: 0.139687
    [34]	valid_0's auc: 0.874887	valid_0's binary_logloss: 0.124169	valid_1's auc: 0.838801	valid_1's binary_logloss: 0.139672
    [35]	valid_0's auc: 0.875447	valid_0's binary_logloss: 0.123934	valid_1's auc: 0.838835	valid_1's binary_logloss: 0.139667
    [36]	valid_0's auc: 0.87617	valid_0's binary_logloss: 0.123693	valid_1's auc: 0.838505	valid_1's binary_logloss: 0.139699
    [37]	valid_0's auc: 0.876793	valid_0's binary_logloss: 0.12346	valid_1's auc: 0.838104	valid_1's binary_logloss: 0.139783
    [38]	valid_0's auc: 0.877265	valid_0's binary_logloss: 0.123251	valid_1's auc: 0.838267	valid_1's binary_logloss: 0.139787
    [39]	valid_0's auc: 0.877869	valid_0's binary_logloss: 0.123018	valid_1's auc: 0.838004	valid_1's binary_logloss: 0.139806
    [40]	valid_0's auc: 0.878509	valid_0's binary_logloss: 0.122803	valid_1's auc: 0.838086	valid_1's binary_logloss: 0.139745
    [41]	valid_0's auc: 0.879077	valid_0's binary_logloss: 0.122585	valid_1's auc: 0.838538	valid_1's binary_logloss: 0.139694
    [42]	valid_0's auc: 0.879515	valid_0's binary_logloss: 0.122368	valid_1's auc: 0.838647	valid_1's binary_logloss: 0.139655
    [43]	valid_0's auc: 0.879985	valid_0's binary_logloss: 0.122166	valid_1's auc: 0.838495	valid_1's binary_logloss: 0.139653
    [44]	valid_0's auc: 0.88041	valid_0's binary_logloss: 0.121985	valid_1's auc: 0.838221	valid_1's binary_logloss: 0.139755
    [45]	valid_0's auc: 0.880907	valid_0's binary_logloss: 0.121777	valid_1's auc: 0.837981	valid_1's binary_logloss: 0.139769
    [46]	valid_0's auc: 0.881216	valid_0's binary_logloss: 0.121594	valid_1's auc: 0.838471	valid_1's binary_logloss: 0.139693
    [47]	valid_0's auc: 0.881591	valid_0's binary_logloss: 0.121422	valid_1's auc: 0.83861	valid_1's binary_logloss: 0.139687
    [48]	valid_0's auc: 0.881867	valid_0's binary_logloss: 0.121266	valid_1's auc: 0.838593	valid_1's binary_logloss: 0.139682
    [49]	valid_0's auc: 0.882285	valid_0's binary_logloss: 0.121041	valid_1's auc: 0.838317	valid_1's binary_logloss: 0.139741
    [50]	valid_0's auc: 0.882828	valid_0's binary_logloss: 0.120853	valid_1's auc: 0.838244	valid_1's binary_logloss: 0.139759
    [51]	valid_0's auc: 0.883154	valid_0's binary_logloss: 0.120688	valid_1's auc: 0.838222	valid_1's binary_logloss: 0.139803
    [52]	valid_0's auc: 0.883348	valid_0's binary_logloss: 0.120567	valid_1's auc: 0.838064	valid_1's binary_logloss: 0.139824
    [53]	valid_0's auc: 0.883583	valid_0's binary_logloss: 0.120424	valid_1's auc: 0.83788	valid_1's binary_logloss: 0.139844
    [54]	valid_0's auc: 0.884106	valid_0's binary_logloss: 0.120208	valid_1's auc: 0.837625	valid_1's binary_logloss: 0.139886
    [55]	valid_0's auc: 0.884777	valid_0's binary_logloss: 0.120039	valid_1's auc: 0.837585	valid_1's binary_logloss: 0.139902
    [56]	valid_0's auc: 0.88511	valid_0's binary_logloss: 0.11989	valid_1's auc: 0.837646	valid_1's binary_logloss: 0.139926
    [57]	valid_0's auc: 0.885365	valid_0's binary_logloss: 0.11975	valid_1's auc: 0.837639	valid_1's binary_logloss: 0.139934
    [58]	valid_0's auc: 0.885606	valid_0's binary_logloss: 0.119595	valid_1's auc: 0.837726	valid_1's binary_logloss: 0.139938
    [59]	valid_0's auc: 0.885965	valid_0's binary_logloss: 0.119403	valid_1's auc: 0.837558	valid_1's binary_logloss: 0.140007
    [60]	valid_0's auc: 0.886208	valid_0's binary_logloss: 0.119263	valid_1's auc: 0.83744	valid_1's binary_logloss: 0.140079
    [61]	valid_0's auc: 0.886458	valid_0's binary_logloss: 0.119118	valid_1's auc: 0.837349	valid_1's binary_logloss: 0.140059
    Early stopping, best iteration is:
    [31]	valid_0's auc: 0.87247	valid_0's binary_logloss: 0.124907	valid_1's auc: 0.838959	valid_1's binary_logloss: 0.139727
    [1]	valid_0's auc: 0.835412	valid_0's binary_logloss: 0.155721	valid_1's auc: 0.81973	valid_1's binary_logloss: 0.164844
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.841188	valid_0's binary_logloss: 0.150354	valid_1's auc: 0.823402	valid_1's binary_logloss: 0.16006
    [3]	valid_0's auc: 0.846758	valid_0's binary_logloss: 0.146288	valid_1's auc: 0.824811	valid_1's binary_logloss: 0.15621
    [4]	valid_0's auc: 0.850398	valid_0's binary_logloss: 0.143088	valid_1's auc: 0.830278	valid_1's binary_logloss: 0.153352
    [5]	valid_0's auc: 0.853086	valid_0's binary_logloss: 0.140514	valid_1's auc: 0.833574	valid_1's binary_logloss: 0.151071
    [6]	valid_0's auc: 0.855915	valid_0's binary_logloss: 0.138329	valid_1's auc: 0.834881	valid_1's binary_logloss: 0.149277
    [7]	valid_0's auc: 0.858115	valid_0's binary_logloss: 0.136481	valid_1's auc: 0.833603	valid_1's binary_logloss: 0.14786
    [8]	valid_0's auc: 0.859479	valid_0's binary_logloss: 0.134947	valid_1's auc: 0.834093	valid_1's binary_logloss: 0.146607
    [9]	valid_0's auc: 0.86143	valid_0's binary_logloss: 0.133519	valid_1's auc: 0.833898	valid_1's binary_logloss: 0.14559
    [10]	valid_0's auc: 0.862964	valid_0's binary_logloss: 0.132331	valid_1's auc: 0.835026	valid_1's binary_logloss: 0.144789
    [11]	valid_0's auc: 0.864277	valid_0's binary_logloss: 0.13126	valid_1's auc: 0.834957	valid_1's binary_logloss: 0.144152
    [12]	valid_0's auc: 0.865572	valid_0's binary_logloss: 0.130304	valid_1's auc: 0.833693	valid_1's binary_logloss: 0.143697
    [13]	valid_0's auc: 0.867519	valid_0's binary_logloss: 0.129385	valid_1's auc: 0.833158	valid_1's binary_logloss: 0.143184
    [14]	valid_0's auc: 0.869354	valid_0's binary_logloss: 0.128524	valid_1's auc: 0.833598	valid_1's binary_logloss: 0.142668
    [15]	valid_0's auc: 0.870553	valid_0's binary_logloss: 0.127746	valid_1's auc: 0.833467	valid_1's binary_logloss: 0.142302
    [16]	valid_0's auc: 0.871816	valid_0's binary_logloss: 0.126943	valid_1's auc: 0.83329	valid_1's binary_logloss: 0.142022
    [17]	valid_0's auc: 0.872964	valid_0's binary_logloss: 0.126266	valid_1's auc: 0.83279	valid_1's binary_logloss: 0.141891
    [18]	valid_0's auc: 0.874047	valid_0's binary_logloss: 0.125646	valid_1's auc: 0.831917	valid_1's binary_logloss: 0.141748
    [19]	valid_0's auc: 0.875336	valid_0's binary_logloss: 0.125072	valid_1's auc: 0.831274	valid_1's binary_logloss: 0.141658
    [20]	valid_0's auc: 0.876959	valid_0's binary_logloss: 0.124484	valid_1's auc: 0.831275	valid_1's binary_logloss: 0.141511
    [21]	valid_0's auc: 0.878049	valid_0's binary_logloss: 0.123928	valid_1's auc: 0.830813	valid_1's binary_logloss: 0.141459
    [22]	valid_0's auc: 0.878905	valid_0's binary_logloss: 0.123447	valid_1's auc: 0.83012	valid_1's binary_logloss: 0.141449
    [23]	valid_0's auc: 0.879827	valid_0's binary_logloss: 0.12295	valid_1's auc: 0.829554	valid_1's binary_logloss: 0.141492
    [24]	valid_0's auc: 0.880692	valid_0's binary_logloss: 0.122479	valid_1's auc: 0.829256	valid_1's binary_logloss: 0.141487
    [25]	valid_0's auc: 0.881715	valid_0's binary_logloss: 0.121994	valid_1's auc: 0.829326	valid_1's binary_logloss: 0.141362
    [26]	valid_0's auc: 0.883014	valid_0's binary_logloss: 0.121527	valid_1's auc: 0.829553	valid_1's binary_logloss: 0.14132
    [27]	valid_0's auc: 0.884245	valid_0's binary_logloss: 0.121024	valid_1's auc: 0.829624	valid_1's binary_logloss: 0.14127
    [28]	valid_0's auc: 0.885238	valid_0's binary_logloss: 0.12058	valid_1's auc: 0.829417	valid_1's binary_logloss: 0.141237
    [29]	valid_0's auc: 0.88602	valid_0's binary_logloss: 0.120198	valid_1's auc: 0.82917	valid_1's binary_logloss: 0.141201
    [30]	valid_0's auc: 0.88684	valid_0's binary_logloss: 0.119831	valid_1's auc: 0.82962	valid_1's binary_logloss: 0.141121
    [31]	valid_0's auc: 0.887965	valid_0's binary_logloss: 0.119437	valid_1's auc: 0.83035	valid_1's binary_logloss: 0.14101
    [32]	valid_0's auc: 0.88868	valid_0's binary_logloss: 0.119086	valid_1's auc: 0.82975	valid_1's binary_logloss: 0.141093
    [33]	valid_0's auc: 0.889895	valid_0's binary_logloss: 0.118649	valid_1's auc: 0.829977	valid_1's binary_logloss: 0.141037
    [34]	valid_0's auc: 0.890626	valid_0's binary_logloss: 0.118328	valid_1's auc: 0.829368	valid_1's binary_logloss: 0.141161
    [35]	valid_0's auc: 0.89116	valid_0's binary_logloss: 0.11806	valid_1's auc: 0.829262	valid_1's binary_logloss: 0.141183
    [36]	valid_0's auc: 0.891999	valid_0's binary_logloss: 0.11775	valid_1's auc: 0.828947	valid_1's binary_logloss: 0.14129
    [37]	valid_0's auc: 0.892306	valid_0's binary_logloss: 0.117477	valid_1's auc: 0.828544	valid_1's binary_logloss: 0.141389
    [38]	valid_0's auc: 0.892937	valid_0's binary_logloss: 0.117192	valid_1's auc: 0.827983	valid_1's binary_logloss: 0.141516
    [39]	valid_0's auc: 0.893563	valid_0's binary_logloss: 0.116869	valid_1's auc: 0.828068	valid_1's binary_logloss: 0.141517
    [40]	valid_0's auc: 0.893942	valid_0's binary_logloss: 0.11662	valid_1's auc: 0.827852	valid_1's binary_logloss: 0.141621
    Early stopping, best iteration is:
    [10]	valid_0's auc: 0.862964	valid_0's binary_logloss: 0.132331	valid_1's auc: 0.835026	valid_1's binary_logloss: 0.144789
    [1]	valid_0's auc: 0.830474	valid_0's binary_logloss: 0.155928	valid_1's auc: 0.817343	valid_1's binary_logloss: 0.164928
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.842931	valid_0's binary_logloss: 0.1503	valid_1's auc: 0.82699	valid_1's binary_logloss: 0.15948
    [3]	valid_0's auc: 0.850877	valid_0's binary_logloss: 0.14631	valid_1's auc: 0.832212	valid_1's binary_logloss: 0.155775
    [4]	valid_0's auc: 0.854431	valid_0's binary_logloss: 0.143104	valid_1's auc: 0.83392	valid_1's binary_logloss: 0.152698
    [5]	valid_0's auc: 0.85663	valid_0's binary_logloss: 0.140582	valid_1's auc: 0.835094	valid_1's binary_logloss: 0.150349
    [6]	valid_0's auc: 0.859142	valid_0's binary_logloss: 0.138289	valid_1's auc: 0.836166	valid_1's binary_logloss: 0.148424
    [7]	valid_0's auc: 0.861364	valid_0's binary_logloss: 0.136413	valid_1's auc: 0.837184	valid_1's binary_logloss: 0.146912
    [8]	valid_0's auc: 0.862199	valid_0's binary_logloss: 0.134841	valid_1's auc: 0.837545	valid_1's binary_logloss: 0.145726
    [9]	valid_0's auc: 0.864095	valid_0's binary_logloss: 0.133364	valid_1's auc: 0.837242	valid_1's binary_logloss: 0.144736
    [10]	valid_0's auc: 0.866024	valid_0's binary_logloss: 0.132096	valid_1's auc: 0.837719	valid_1's binary_logloss: 0.143766
    [11]	valid_0's auc: 0.867454	valid_0's binary_logloss: 0.131002	valid_1's auc: 0.837865	valid_1's binary_logloss: 0.143009
    [12]	valid_0's auc: 0.868329	valid_0's binary_logloss: 0.130024	valid_1's auc: 0.837259	valid_1's binary_logloss: 0.14244
    [13]	valid_0's auc: 0.869137	valid_0's binary_logloss: 0.129145	valid_1's auc: 0.837689	valid_1's binary_logloss: 0.141896
    [14]	valid_0's auc: 0.870957	valid_0's binary_logloss: 0.128226	valid_1's auc: 0.838226	valid_1's binary_logloss: 0.141392
    [15]	valid_0's auc: 0.872273	valid_0's binary_logloss: 0.12745	valid_1's auc: 0.837906	valid_1's binary_logloss: 0.141019
    [16]	valid_0's auc: 0.873243	valid_0's binary_logloss: 0.12672	valid_1's auc: 0.837761	valid_1's binary_logloss: 0.140677
    [17]	valid_0's auc: 0.874251	valid_0's binary_logloss: 0.126044	valid_1's auc: 0.83701	valid_1's binary_logloss: 0.140582
    [18]	valid_0's auc: 0.875622	valid_0's binary_logloss: 0.125387	valid_1's auc: 0.836179	valid_1's binary_logloss: 0.140485
    [19]	valid_0's auc: 0.877031	valid_0's binary_logloss: 0.124759	valid_1's auc: 0.836188	valid_1's binary_logloss: 0.14029
    [20]	valid_0's auc: 0.878046	valid_0's binary_logloss: 0.124156	valid_1's auc: 0.836531	valid_1's binary_logloss: 0.140133
    [21]	valid_0's auc: 0.879478	valid_0's binary_logloss: 0.123507	valid_1's auc: 0.837068	valid_1's binary_logloss: 0.13995
    [22]	valid_0's auc: 0.880423	valid_0's binary_logloss: 0.123029	valid_1's auc: 0.836817	valid_1's binary_logloss: 0.139912
    [23]	valid_0's auc: 0.881684	valid_0's binary_logloss: 0.122492	valid_1's auc: 0.836983	valid_1's binary_logloss: 0.139762
    [24]	valid_0's auc: 0.882873	valid_0's binary_logloss: 0.121986	valid_1's auc: 0.837319	valid_1's binary_logloss: 0.139659
    [25]	valid_0's auc: 0.883597	valid_0's binary_logloss: 0.121566	valid_1's auc: 0.837154	valid_1's binary_logloss: 0.139623
    [26]	valid_0's auc: 0.884814	valid_0's binary_logloss: 0.121104	valid_1's auc: 0.836302	valid_1's binary_logloss: 0.139668
    [27]	valid_0's auc: 0.886026	valid_0's binary_logloss: 0.120635	valid_1's auc: 0.836521	valid_1's binary_logloss: 0.139601
    [28]	valid_0's auc: 0.887071	valid_0's binary_logloss: 0.120222	valid_1's auc: 0.836646	valid_1's binary_logloss: 0.139557
    [29]	valid_0's auc: 0.887946	valid_0's binary_logloss: 0.119804	valid_1's auc: 0.836735	valid_1's binary_logloss: 0.139518
    [30]	valid_0's auc: 0.88898	valid_0's binary_logloss: 0.119416	valid_1's auc: 0.836858	valid_1's binary_logloss: 0.139499
    [31]	valid_0's auc: 0.889792	valid_0's binary_logloss: 0.119058	valid_1's auc: 0.836917	valid_1's binary_logloss: 0.139463
    [32]	valid_0's auc: 0.890876	valid_0's binary_logloss: 0.118631	valid_1's auc: 0.836346	valid_1's binary_logloss: 0.139532
    [33]	valid_0's auc: 0.891629	valid_0's binary_logloss: 0.118259	valid_1's auc: 0.836206	valid_1's binary_logloss: 0.139603
    [34]	valid_0's auc: 0.892446	valid_0's binary_logloss: 0.117893	valid_1's auc: 0.836005	valid_1's binary_logloss: 0.139603
    [35]	valid_0's auc: 0.893407	valid_0's binary_logloss: 0.11752	valid_1's auc: 0.8361	valid_1's binary_logloss: 0.139574
    [36]	valid_0's auc: 0.893836	valid_0's binary_logloss: 0.117247	valid_1's auc: 0.836147	valid_1's binary_logloss: 0.139608
    [37]	valid_0's auc: 0.894774	valid_0's binary_logloss: 0.116913	valid_1's auc: 0.836601	valid_1's binary_logloss: 0.139569
    [38]	valid_0's auc: 0.895494	valid_0's binary_logloss: 0.116611	valid_1's auc: 0.836232	valid_1's binary_logloss: 0.139645
    [39]	valid_0's auc: 0.896102	valid_0's binary_logloss: 0.116275	valid_1's auc: 0.836415	valid_1's binary_logloss: 0.139653
    [40]	valid_0's auc: 0.896715	valid_0's binary_logloss: 0.115934	valid_1's auc: 0.836463	valid_1's binary_logloss: 0.139671
    [41]	valid_0's auc: 0.897232	valid_0's binary_logloss: 0.115612	valid_1's auc: 0.836223	valid_1's binary_logloss: 0.139762
    [42]	valid_0's auc: 0.897875	valid_0's binary_logloss: 0.11528	valid_1's auc: 0.836151	valid_1's binary_logloss: 0.139777
    [43]	valid_0's auc: 0.898493	valid_0's binary_logloss: 0.114999	valid_1's auc: 0.836216	valid_1's binary_logloss: 0.139761
    [44]	valid_0's auc: 0.899179	valid_0's binary_logloss: 0.114703	valid_1's auc: 0.836328	valid_1's binary_logloss: 0.139755
    Early stopping, best iteration is:
    [14]	valid_0's auc: 0.870957	valid_0's binary_logloss: 0.128226	valid_1's auc: 0.838226	valid_1's binary_logloss: 0.141392
    [1]	valid_0's auc: 0.834724	valid_0's binary_logloss: 0.15607	valid_1's auc: 0.822983	valid_1's binary_logloss: 0.165104
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.842835	valid_0's binary_logloss: 0.150494	valid_1's auc: 0.830472	valid_1's binary_logloss: 0.159671
    [3]	valid_0's auc: 0.847187	valid_0's binary_logloss: 0.146306	valid_1's auc: 0.830873	valid_1's binary_logloss: 0.155985
    [4]	valid_0's auc: 0.850394	valid_0's binary_logloss: 0.143088	valid_1's auc: 0.830975	valid_1's binary_logloss: 0.15321
    [5]	valid_0's auc: 0.853379	valid_0's binary_logloss: 0.140508	valid_1's auc: 0.832135	valid_1's binary_logloss: 0.150854
    [6]	valid_0's auc: 0.855463	valid_0's binary_logloss: 0.138297	valid_1's auc: 0.833116	valid_1's binary_logloss: 0.149013
    [7]	valid_0's auc: 0.856723	valid_0's binary_logloss: 0.136504	valid_1's auc: 0.833811	valid_1's binary_logloss: 0.147577
    [8]	valid_0's auc: 0.858076	valid_0's binary_logloss: 0.13495	valid_1's auc: 0.835315	valid_1's binary_logloss: 0.146273
    [9]	valid_0's auc: 0.861024	valid_0's binary_logloss: 0.133583	valid_1's auc: 0.835042	valid_1's binary_logloss: 0.145374
    [10]	valid_0's auc: 0.862281	valid_0's binary_logloss: 0.132357	valid_1's auc: 0.834154	valid_1's binary_logloss: 0.144649
    [11]	valid_0's auc: 0.864612	valid_0's binary_logloss: 0.131283	valid_1's auc: 0.834587	valid_1's binary_logloss: 0.143941
    [12]	valid_0's auc: 0.866377	valid_0's binary_logloss: 0.130299	valid_1's auc: 0.834242	valid_1's binary_logloss: 0.143366
    [13]	valid_0's auc: 0.868343	valid_0's binary_logloss: 0.129417	valid_1's auc: 0.833273	valid_1's binary_logloss: 0.142976
    [14]	valid_0's auc: 0.86957	valid_0's binary_logloss: 0.128593	valid_1's auc: 0.833783	valid_1's binary_logloss: 0.142567
    [15]	valid_0's auc: 0.871109	valid_0's binary_logloss: 0.127759	valid_1's auc: 0.834057	valid_1's binary_logloss: 0.142234
    [16]	valid_0's auc: 0.872893	valid_0's binary_logloss: 0.126996	valid_1's auc: 0.835329	valid_1's binary_logloss: 0.141809
    [17]	valid_0's auc: 0.874236	valid_0's binary_logloss: 0.12631	valid_1's auc: 0.834985	valid_1's binary_logloss: 0.141613
    [18]	valid_0's auc: 0.875324	valid_0's binary_logloss: 0.125725	valid_1's auc: 0.834942	valid_1's binary_logloss: 0.141363
    [19]	valid_0's auc: 0.876659	valid_0's binary_logloss: 0.125068	valid_1's auc: 0.835024	valid_1's binary_logloss: 0.141162
    [20]	valid_0's auc: 0.877885	valid_0's binary_logloss: 0.124484	valid_1's auc: 0.835893	valid_1's binary_logloss: 0.140933
    [21]	valid_0's auc: 0.879121	valid_0's binary_logloss: 0.12391	valid_1's auc: 0.837029	valid_1's binary_logloss: 0.140651
    [22]	valid_0's auc: 0.880116	valid_0's binary_logloss: 0.123339	valid_1's auc: 0.837366	valid_1's binary_logloss: 0.140547
    [23]	valid_0's auc: 0.881224	valid_0's binary_logloss: 0.12282	valid_1's auc: 0.837357	valid_1's binary_logloss: 0.140445
    [24]	valid_0's auc: 0.882014	valid_0's binary_logloss: 0.122386	valid_1's auc: 0.837343	valid_1's binary_logloss: 0.140371
    [25]	valid_0's auc: 0.88318	valid_0's binary_logloss: 0.121861	valid_1's auc: 0.83723	valid_1's binary_logloss: 0.140313
    [26]	valid_0's auc: 0.884008	valid_0's binary_logloss: 0.121441	valid_1's auc: 0.837761	valid_1's binary_logloss: 0.140173
    [27]	valid_0's auc: 0.884676	valid_0's binary_logloss: 0.121001	valid_1's auc: 0.838046	valid_1's binary_logloss: 0.140086
    [28]	valid_0's auc: 0.885524	valid_0's binary_logloss: 0.120598	valid_1's auc: 0.838029	valid_1's binary_logloss: 0.140051
    [29]	valid_0's auc: 0.886461	valid_0's binary_logloss: 0.120157	valid_1's auc: 0.837775	valid_1's binary_logloss: 0.140057
    [30]	valid_0's auc: 0.887053	valid_0's binary_logloss: 0.119807	valid_1's auc: 0.837472	valid_1's binary_logloss: 0.140111
    [31]	valid_0's auc: 0.888177	valid_0's binary_logloss: 0.119425	valid_1's auc: 0.837575	valid_1's binary_logloss: 0.140093
    [32]	valid_0's auc: 0.889072	valid_0's binary_logloss: 0.119055	valid_1's auc: 0.837158	valid_1's binary_logloss: 0.140195
    [33]	valid_0's auc: 0.889782	valid_0's binary_logloss: 0.118676	valid_1's auc: 0.837296	valid_1's binary_logloss: 0.140221
    [34]	valid_0's auc: 0.890876	valid_0's binary_logloss: 0.118304	valid_1's auc: 0.837481	valid_1's binary_logloss: 0.140165
    [35]	valid_0's auc: 0.891448	valid_0's binary_logloss: 0.11798	valid_1's auc: 0.837953	valid_1's binary_logloss: 0.140085
    [36]	valid_0's auc: 0.892165	valid_0's binary_logloss: 0.11764	valid_1's auc: 0.837794	valid_1's binary_logloss: 0.140112
    [37]	valid_0's auc: 0.892798	valid_0's binary_logloss: 0.117321	valid_1's auc: 0.837291	valid_1's binary_logloss: 0.140221
    [38]	valid_0's auc: 0.893318	valid_0's binary_logloss: 0.117028	valid_1's auc: 0.837278	valid_1's binary_logloss: 0.140221
    [39]	valid_0's auc: 0.894018	valid_0's binary_logloss: 0.116742	valid_1's auc: 0.83724	valid_1's binary_logloss: 0.140232
    [40]	valid_0's auc: 0.894781	valid_0's binary_logloss: 0.116373	valid_1's auc: 0.836901	valid_1's binary_logloss: 0.140328
    [41]	valid_0's auc: 0.895222	valid_0's binary_logloss: 0.116075	valid_1's auc: 0.836655	valid_1's binary_logloss: 0.140422
    [42]	valid_0's auc: 0.895842	valid_0's binary_logloss: 0.115755	valid_1's auc: 0.836383	valid_1's binary_logloss: 0.140503
    [43]	valid_0's auc: 0.896389	valid_0's binary_logloss: 0.115503	valid_1's auc: 0.836348	valid_1's binary_logloss: 0.140505
    [44]	valid_0's auc: 0.896843	valid_0's binary_logloss: 0.115204	valid_1's auc: 0.836521	valid_1's binary_logloss: 0.140518
    [45]	valid_0's auc: 0.897272	valid_0's binary_logloss: 0.114886	valid_1's auc: 0.836311	valid_1's binary_logloss: 0.140581
    [46]	valid_0's auc: 0.898034	valid_0's binary_logloss: 0.114544	valid_1's auc: 0.835871	valid_1's binary_logloss: 0.140663
    [47]	valid_0's auc: 0.898562	valid_0's binary_logloss: 0.114262	valid_1's auc: 0.835926	valid_1's binary_logloss: 0.140642
    [48]	valid_0's auc: 0.898919	valid_0's binary_logloss: 0.114006	valid_1's auc: 0.835849	valid_1's binary_logloss: 0.140687
    [49]	valid_0's auc: 0.899111	valid_0's binary_logloss: 0.113791	valid_1's auc: 0.835874	valid_1's binary_logloss: 0.140728
    [50]	valid_0's auc: 0.89987	valid_0's binary_logloss: 0.113543	valid_1's auc: 0.835915	valid_1's binary_logloss: 0.14075
    [51]	valid_0's auc: 0.90004	valid_0's binary_logloss: 0.113342	valid_1's auc: 0.835947	valid_1's binary_logloss: 0.140748
    [52]	valid_0's auc: 0.900405	valid_0's binary_logloss: 0.113087	valid_1's auc: 0.836011	valid_1's binary_logloss: 0.140767
    [53]	valid_0's auc: 0.900828	valid_0's binary_logloss: 0.112831	valid_1's auc: 0.836259	valid_1's binary_logloss: 0.140771
    [54]	valid_0's auc: 0.901597	valid_0's binary_logloss: 0.112604	valid_1's auc: 0.836296	valid_1's binary_logloss: 0.14078
    [55]	valid_0's auc: 0.901645	valid_0's binary_logloss: 0.112429	valid_1's auc: 0.836095	valid_1's binary_logloss: 0.140822
    [56]	valid_0's auc: 0.902162	valid_0's binary_logloss: 0.112169	valid_1's auc: 0.835965	valid_1's binary_logloss: 0.14086
    [57]	valid_0's auc: 0.902422	valid_0's binary_logloss: 0.111944	valid_1's auc: 0.835493	valid_1's binary_logloss: 0.140993
    Early stopping, best iteration is:
    [27]	valid_0's auc: 0.884676	valid_0's binary_logloss: 0.121001	valid_1's auc: 0.838046	valid_1's binary_logloss: 0.140086
    [1]	valid_0's auc: 0.835412	valid_0's binary_logloss: 0.155721	valid_1's auc: 0.81973	valid_1's binary_logloss: 0.164844
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.841188	valid_0's binary_logloss: 0.150354	valid_1's auc: 0.823402	valid_1's binary_logloss: 0.16006
    [3]	valid_0's auc: 0.846758	valid_0's binary_logloss: 0.146288	valid_1's auc: 0.824811	valid_1's binary_logloss: 0.15621
    [4]	valid_0's auc: 0.850398	valid_0's binary_logloss: 0.143088	valid_1's auc: 0.830278	valid_1's binary_logloss: 0.153352
    [5]	valid_0's auc: 0.853086	valid_0's binary_logloss: 0.140514	valid_1's auc: 0.833574	valid_1's binary_logloss: 0.151071
    [6]	valid_0's auc: 0.855915	valid_0's binary_logloss: 0.138329	valid_1's auc: 0.834881	valid_1's binary_logloss: 0.149277
    [7]	valid_0's auc: 0.858115	valid_0's binary_logloss: 0.136481	valid_1's auc: 0.833603	valid_1's binary_logloss: 0.14786
    [8]	valid_0's auc: 0.859479	valid_0's binary_logloss: 0.134947	valid_1's auc: 0.834093	valid_1's binary_logloss: 0.146607
    [9]	valid_0's auc: 0.86143	valid_0's binary_logloss: 0.133519	valid_1's auc: 0.833898	valid_1's binary_logloss: 0.14559
    [10]	valid_0's auc: 0.862964	valid_0's binary_logloss: 0.132331	valid_1's auc: 0.835026	valid_1's binary_logloss: 0.144789
    [11]	valid_0's auc: 0.864277	valid_0's binary_logloss: 0.13126	valid_1's auc: 0.834957	valid_1's binary_logloss: 0.144152
    [12]	valid_0's auc: 0.865572	valid_0's binary_logloss: 0.130304	valid_1's auc: 0.833693	valid_1's binary_logloss: 0.143697
    [13]	valid_0's auc: 0.867519	valid_0's binary_logloss: 0.129385	valid_1's auc: 0.833158	valid_1's binary_logloss: 0.143184
    [14]	valid_0's auc: 0.869354	valid_0's binary_logloss: 0.128524	valid_1's auc: 0.833598	valid_1's binary_logloss: 0.142668
    [15]	valid_0's auc: 0.870553	valid_0's binary_logloss: 0.127746	valid_1's auc: 0.833467	valid_1's binary_logloss: 0.142302
    [16]	valid_0's auc: 0.871816	valid_0's binary_logloss: 0.126943	valid_1's auc: 0.83329	valid_1's binary_logloss: 0.142022
    [17]	valid_0's auc: 0.872964	valid_0's binary_logloss: 0.126266	valid_1's auc: 0.83279	valid_1's binary_logloss: 0.141891
    [18]	valid_0's auc: 0.874047	valid_0's binary_logloss: 0.125646	valid_1's auc: 0.831917	valid_1's binary_logloss: 0.141748
    [19]	valid_0's auc: 0.875336	valid_0's binary_logloss: 0.125072	valid_1's auc: 0.831274	valid_1's binary_logloss: 0.141658
    [20]	valid_0's auc: 0.876959	valid_0's binary_logloss: 0.124484	valid_1's auc: 0.831275	valid_1's binary_logloss: 0.141511
    [21]	valid_0's auc: 0.878049	valid_0's binary_logloss: 0.123928	valid_1's auc: 0.830813	valid_1's binary_logloss: 0.141459
    [22]	valid_0's auc: 0.878905	valid_0's binary_logloss: 0.123447	valid_1's auc: 0.83012	valid_1's binary_logloss: 0.141449
    [23]	valid_0's auc: 0.879827	valid_0's binary_logloss: 0.12295	valid_1's auc: 0.829554	valid_1's binary_logloss: 0.141492
    [24]	valid_0's auc: 0.880692	valid_0's binary_logloss: 0.122479	valid_1's auc: 0.829256	valid_1's binary_logloss: 0.141487
    [25]	valid_0's auc: 0.881715	valid_0's binary_logloss: 0.121994	valid_1's auc: 0.829326	valid_1's binary_logloss: 0.141362
    [26]	valid_0's auc: 0.883014	valid_0's binary_logloss: 0.121527	valid_1's auc: 0.829553	valid_1's binary_logloss: 0.14132
    [27]	valid_0's auc: 0.884245	valid_0's binary_logloss: 0.121024	valid_1's auc: 0.829624	valid_1's binary_logloss: 0.14127
    [28]	valid_0's auc: 0.885238	valid_0's binary_logloss: 0.12058	valid_1's auc: 0.829417	valid_1's binary_logloss: 0.141237
    [29]	valid_0's auc: 0.88602	valid_0's binary_logloss: 0.120198	valid_1's auc: 0.82917	valid_1's binary_logloss: 0.141201
    [30]	valid_0's auc: 0.88684	valid_0's binary_logloss: 0.119831	valid_1's auc: 0.82962	valid_1's binary_logloss: 0.141121
    [31]	valid_0's auc: 0.887965	valid_0's binary_logloss: 0.119437	valid_1's auc: 0.83035	valid_1's binary_logloss: 0.14101
    [32]	valid_0's auc: 0.88868	valid_0's binary_logloss: 0.119086	valid_1's auc: 0.82975	valid_1's binary_logloss: 0.141093
    [33]	valid_0's auc: 0.889895	valid_0's binary_logloss: 0.118649	valid_1's auc: 0.829977	valid_1's binary_logloss: 0.141037
    [34]	valid_0's auc: 0.890626	valid_0's binary_logloss: 0.118328	valid_1's auc: 0.829368	valid_1's binary_logloss: 0.141161
    [35]	valid_0's auc: 0.89116	valid_0's binary_logloss: 0.11806	valid_1's auc: 0.829262	valid_1's binary_logloss: 0.141183
    [36]	valid_0's auc: 0.891999	valid_0's binary_logloss: 0.11775	valid_1's auc: 0.828947	valid_1's binary_logloss: 0.14129
    [37]	valid_0's auc: 0.892306	valid_0's binary_logloss: 0.117477	valid_1's auc: 0.828544	valid_1's binary_logloss: 0.141389
    [38]	valid_0's auc: 0.892937	valid_0's binary_logloss: 0.117192	valid_1's auc: 0.827983	valid_1's binary_logloss: 0.141516
    [39]	valid_0's auc: 0.893563	valid_0's binary_logloss: 0.116869	valid_1's auc: 0.828068	valid_1's binary_logloss: 0.141517
    [40]	valid_0's auc: 0.893942	valid_0's binary_logloss: 0.11662	valid_1's auc: 0.827852	valid_1's binary_logloss: 0.141621
    Early stopping, best iteration is:
    [10]	valid_0's auc: 0.862964	valid_0's binary_logloss: 0.132331	valid_1's auc: 0.835026	valid_1's binary_logloss: 0.144789
    [1]	valid_0's auc: 0.830474	valid_0's binary_logloss: 0.155928	valid_1's auc: 0.817343	valid_1's binary_logloss: 0.164928
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.842931	valid_0's binary_logloss: 0.1503	valid_1's auc: 0.82699	valid_1's binary_logloss: 0.15948
    [3]	valid_0's auc: 0.850877	valid_0's binary_logloss: 0.14631	valid_1's auc: 0.832212	valid_1's binary_logloss: 0.155775
    [4]	valid_0's auc: 0.854431	valid_0's binary_logloss: 0.143104	valid_1's auc: 0.83392	valid_1's binary_logloss: 0.152698
    [5]	valid_0's auc: 0.85663	valid_0's binary_logloss: 0.140582	valid_1's auc: 0.835094	valid_1's binary_logloss: 0.150349
    [6]	valid_0's auc: 0.859142	valid_0's binary_logloss: 0.138289	valid_1's auc: 0.836166	valid_1's binary_logloss: 0.148424
    [7]	valid_0's auc: 0.861364	valid_0's binary_logloss: 0.136413	valid_1's auc: 0.837184	valid_1's binary_logloss: 0.146912
    [8]	valid_0's auc: 0.862199	valid_0's binary_logloss: 0.134841	valid_1's auc: 0.837545	valid_1's binary_logloss: 0.145726
    [9]	valid_0's auc: 0.864095	valid_0's binary_logloss: 0.133364	valid_1's auc: 0.837242	valid_1's binary_logloss: 0.144736
    [10]	valid_0's auc: 0.866024	valid_0's binary_logloss: 0.132096	valid_1's auc: 0.837719	valid_1's binary_logloss: 0.143766
    [11]	valid_0's auc: 0.867454	valid_0's binary_logloss: 0.131002	valid_1's auc: 0.837865	valid_1's binary_logloss: 0.143009
    [12]	valid_0's auc: 0.868329	valid_0's binary_logloss: 0.130024	valid_1's auc: 0.837259	valid_1's binary_logloss: 0.14244
    [13]	valid_0's auc: 0.869137	valid_0's binary_logloss: 0.129145	valid_1's auc: 0.837689	valid_1's binary_logloss: 0.141896
    [14]	valid_0's auc: 0.870957	valid_0's binary_logloss: 0.128226	valid_1's auc: 0.838226	valid_1's binary_logloss: 0.141392
    [15]	valid_0's auc: 0.872273	valid_0's binary_logloss: 0.12745	valid_1's auc: 0.837906	valid_1's binary_logloss: 0.141019
    [16]	valid_0's auc: 0.873243	valid_0's binary_logloss: 0.12672	valid_1's auc: 0.837761	valid_1's binary_logloss: 0.140677
    [17]	valid_0's auc: 0.874251	valid_0's binary_logloss: 0.126044	valid_1's auc: 0.83701	valid_1's binary_logloss: 0.140582
    [18]	valid_0's auc: 0.875622	valid_0's binary_logloss: 0.125387	valid_1's auc: 0.836179	valid_1's binary_logloss: 0.140485
    [19]	valid_0's auc: 0.877031	valid_0's binary_logloss: 0.124759	valid_1's auc: 0.836188	valid_1's binary_logloss: 0.14029
    [20]	valid_0's auc: 0.878046	valid_0's binary_logloss: 0.124156	valid_1's auc: 0.836531	valid_1's binary_logloss: 0.140133
    [21]	valid_0's auc: 0.879478	valid_0's binary_logloss: 0.123507	valid_1's auc: 0.837068	valid_1's binary_logloss: 0.13995
    [22]	valid_0's auc: 0.880423	valid_0's binary_logloss: 0.123029	valid_1's auc: 0.836817	valid_1's binary_logloss: 0.139912
    [23]	valid_0's auc: 0.881684	valid_0's binary_logloss: 0.122492	valid_1's auc: 0.836983	valid_1's binary_logloss: 0.139762
    [24]	valid_0's auc: 0.882873	valid_0's binary_logloss: 0.121986	valid_1's auc: 0.837319	valid_1's binary_logloss: 0.139659
    [25]	valid_0's auc: 0.883597	valid_0's binary_logloss: 0.121566	valid_1's auc: 0.837154	valid_1's binary_logloss: 0.139623
    [26]	valid_0's auc: 0.884814	valid_0's binary_logloss: 0.121104	valid_1's auc: 0.836302	valid_1's binary_logloss: 0.139668
    [27]	valid_0's auc: 0.886026	valid_0's binary_logloss: 0.120635	valid_1's auc: 0.836521	valid_1's binary_logloss: 0.139601
    [28]	valid_0's auc: 0.887071	valid_0's binary_logloss: 0.120222	valid_1's auc: 0.836646	valid_1's binary_logloss: 0.139557
    [29]	valid_0's auc: 0.887946	valid_0's binary_logloss: 0.119804	valid_1's auc: 0.836735	valid_1's binary_logloss: 0.139518
    [30]	valid_0's auc: 0.88898	valid_0's binary_logloss: 0.119416	valid_1's auc: 0.836858	valid_1's binary_logloss: 0.139499
    [31]	valid_0's auc: 0.889792	valid_0's binary_logloss: 0.119058	valid_1's auc: 0.836917	valid_1's binary_logloss: 0.139463
    [32]	valid_0's auc: 0.890876	valid_0's binary_logloss: 0.118631	valid_1's auc: 0.836346	valid_1's binary_logloss: 0.139532
    [33]	valid_0's auc: 0.891629	valid_0's binary_logloss: 0.118259	valid_1's auc: 0.836206	valid_1's binary_logloss: 0.139603
    [34]	valid_0's auc: 0.892446	valid_0's binary_logloss: 0.117893	valid_1's auc: 0.836005	valid_1's binary_logloss: 0.139603
    [35]	valid_0's auc: 0.893407	valid_0's binary_logloss: 0.11752	valid_1's auc: 0.8361	valid_1's binary_logloss: 0.139574
    [36]	valid_0's auc: 0.893836	valid_0's binary_logloss: 0.117247	valid_1's auc: 0.836147	valid_1's binary_logloss: 0.139608
    [37]	valid_0's auc: 0.894774	valid_0's binary_logloss: 0.116913	valid_1's auc: 0.836601	valid_1's binary_logloss: 0.139569
    [38]	valid_0's auc: 0.895494	valid_0's binary_logloss: 0.116611	valid_1's auc: 0.836232	valid_1's binary_logloss: 0.139645
    [39]	valid_0's auc: 0.896102	valid_0's binary_logloss: 0.116275	valid_1's auc: 0.836415	valid_1's binary_logloss: 0.139653
    [40]	valid_0's auc: 0.896715	valid_0's binary_logloss: 0.115934	valid_1's auc: 0.836463	valid_1's binary_logloss: 0.139671
    [41]	valid_0's auc: 0.897232	valid_0's binary_logloss: 0.115612	valid_1's auc: 0.836223	valid_1's binary_logloss: 0.139762
    [42]	valid_0's auc: 0.897875	valid_0's binary_logloss: 0.11528	valid_1's auc: 0.836151	valid_1's binary_logloss: 0.139777
    [43]	valid_0's auc: 0.898493	valid_0's binary_logloss: 0.114999	valid_1's auc: 0.836216	valid_1's binary_logloss: 0.139761
    [44]	valid_0's auc: 0.899179	valid_0's binary_logloss: 0.114703	valid_1's auc: 0.836328	valid_1's binary_logloss: 0.139755
    Early stopping, best iteration is:
    [14]	valid_0's auc: 0.870957	valid_0's binary_logloss: 0.128226	valid_1's auc: 0.838226	valid_1's binary_logloss: 0.141392
    [1]	valid_0's auc: 0.834724	valid_0's binary_logloss: 0.15607	valid_1's auc: 0.822983	valid_1's binary_logloss: 0.165104
    Training until validation scores don't improve for 30 rounds
    [2]	valid_0's auc: 0.842835	valid_0's binary_logloss: 0.150494	valid_1's auc: 0.830472	valid_1's binary_logloss: 0.159671
    [3]	valid_0's auc: 0.847187	valid_0's binary_logloss: 0.146306	valid_1's auc: 0.830873	valid_1's binary_logloss: 0.155985
    [4]	valid_0's auc: 0.850394	valid_0's binary_logloss: 0.143088	valid_1's auc: 0.830975	valid_1's binary_logloss: 0.15321
    [5]	valid_0's auc: 0.853379	valid_0's binary_logloss: 0.140508	valid_1's auc: 0.832135	valid_1's binary_logloss: 0.150854
    [6]	valid_0's auc: 0.855463	valid_0's binary_logloss: 0.138297	valid_1's auc: 0.833116	valid_1's binary_logloss: 0.149013
    [7]	valid_0's auc: 0.856723	valid_0's binary_logloss: 0.136504	valid_1's auc: 0.833811	valid_1's binary_logloss: 0.147577
    [8]	valid_0's auc: 0.858076	valid_0's binary_logloss: 0.13495	valid_1's auc: 0.835315	valid_1's binary_logloss: 0.146273
    [9]	valid_0's auc: 0.861024	valid_0's binary_logloss: 0.133583	valid_1's auc: 0.835042	valid_1's binary_logloss: 0.145374
    [10]	valid_0's auc: 0.862281	valid_0's binary_logloss: 0.132357	valid_1's auc: 0.834154	valid_1's binary_logloss: 0.144649
    [11]	valid_0's auc: 0.864612	valid_0's binary_logloss: 0.131283	valid_1's auc: 0.834587	valid_1's binary_logloss: 0.143941
    [12]	valid_0's auc: 0.866377	valid_0's binary_logloss: 0.130299	valid_1's auc: 0.834242	valid_1's binary_logloss: 0.143366
    [13]	valid_0's auc: 0.868343	valid_0's binary_logloss: 0.129417	valid_1's auc: 0.833273	valid_1's binary_logloss: 0.142976
    [14]	valid_0's auc: 0.86957	valid_0's binary_logloss: 0.128593	valid_1's auc: 0.833783	valid_1's binary_logloss: 0.142567
    [15]	valid_0's auc: 0.871109	valid_0's binary_logloss: 0.127759	valid_1's auc: 0.834057	valid_1's binary_logloss: 0.142234
    [16]	valid_0's auc: 0.872893	valid_0's binary_logloss: 0.126996	valid_1's auc: 0.835329	valid_1's binary_logloss: 0.141809
    [17]	valid_0's auc: 0.874236	valid_0's binary_logloss: 0.12631	valid_1's auc: 0.834985	valid_1's binary_logloss: 0.141613
    [18]	valid_0's auc: 0.875324	valid_0's binary_logloss: 0.125725	valid_1's auc: 0.834942	valid_1's binary_logloss: 0.141363
    [19]	valid_0's auc: 0.876659	valid_0's binary_logloss: 0.125068	valid_1's auc: 0.835024	valid_1's binary_logloss: 0.141162
    [20]	valid_0's auc: 0.877885	valid_0's binary_logloss: 0.124484	valid_1's auc: 0.835893	valid_1's binary_logloss: 0.140933
    [21]	valid_0's auc: 0.879121	valid_0's binary_logloss: 0.12391	valid_1's auc: 0.837029	valid_1's binary_logloss: 0.140651
    [22]	valid_0's auc: 0.880116	valid_0's binary_logloss: 0.123339	valid_1's auc: 0.837366	valid_1's binary_logloss: 0.140547
    [23]	valid_0's auc: 0.881224	valid_0's binary_logloss: 0.12282	valid_1's auc: 0.837357	valid_1's binary_logloss: 0.140445
    [24]	valid_0's auc: 0.882014	valid_0's binary_logloss: 0.122386	valid_1's auc: 0.837343	valid_1's binary_logloss: 0.140371
    [25]	valid_0's auc: 0.88318	valid_0's binary_logloss: 0.121861	valid_1's auc: 0.83723	valid_1's binary_logloss: 0.140313
    [26]	valid_0's auc: 0.884008	valid_0's binary_logloss: 0.121441	valid_1's auc: 0.837761	valid_1's binary_logloss: 0.140173
    [27]	valid_0's auc: 0.884676	valid_0's binary_logloss: 0.121001	valid_1's auc: 0.838046	valid_1's binary_logloss: 0.140086
    [28]	valid_0's auc: 0.885524	valid_0's binary_logloss: 0.120598	valid_1's auc: 0.838029	valid_1's binary_logloss: 0.140051
    [29]	valid_0's auc: 0.886461	valid_0's binary_logloss: 0.120157	valid_1's auc: 0.837775	valid_1's binary_logloss: 0.140057
    [30]	valid_0's auc: 0.887053	valid_0's binary_logloss: 0.119807	valid_1's auc: 0.837472	valid_1's binary_logloss: 0.140111
    [31]	valid_0's auc: 0.888177	valid_0's binary_logloss: 0.119425	valid_1's auc: 0.837575	valid_1's binary_logloss: 0.140093
    [32]	valid_0's auc: 0.889072	valid_0's binary_logloss: 0.119055	valid_1's auc: 0.837158	valid_1's binary_logloss: 0.140195
    [33]	valid_0's auc: 0.889782	valid_0's binary_logloss: 0.118676	valid_1's auc: 0.837296	valid_1's binary_logloss: 0.140221
    [34]	valid_0's auc: 0.890876	valid_0's binary_logloss: 0.118304	valid_1's auc: 0.837481	valid_1's binary_logloss: 0.140165
    [35]	valid_0's auc: 0.891448	valid_0's binary_logloss: 0.11798	valid_1's auc: 0.837953	valid_1's binary_logloss: 0.140085
    [36]	valid_0's auc: 0.892165	valid_0's binary_logloss: 0.11764	valid_1's auc: 0.837794	valid_1's binary_logloss: 0.140112
    [37]	valid_0's auc: 0.892798	valid_0's binary_logloss: 0.117321	valid_1's auc: 0.837291	valid_1's binary_logloss: 0.140221
    [38]	valid_0's auc: 0.893318	valid_0's binary_logloss: 0.117028	valid_1's auc: 0.837278	valid_1's binary_logloss: 0.140221
    [39]	valid_0's auc: 0.894018	valid_0's binary_logloss: 0.116742	valid_1's auc: 0.83724	valid_1's binary_logloss: 0.140232
    [40]	valid_0's auc: 0.894781	valid_0's binary_logloss: 0.116373	valid_1's auc: 0.836901	valid_1's binary_logloss: 0.140328
    [41]	valid_0's auc: 0.895222	valid_0's binary_logloss: 0.116075	valid_1's auc: 0.836655	valid_1's binary_logloss: 0.140422
    [42]	valid_0's auc: 0.895842	valid_0's binary_logloss: 0.115755	valid_1's auc: 0.836383	valid_1's binary_logloss: 0.140503
    [43]	valid_0's auc: 0.896389	valid_0's binary_logloss: 0.115503	valid_1's auc: 0.836348	valid_1's binary_logloss: 0.140505
    [44]	valid_0's auc: 0.896843	valid_0's binary_logloss: 0.115204	valid_1's auc: 0.836521	valid_1's binary_logloss: 0.140518
    [45]	valid_0's auc: 0.897272	valid_0's binary_logloss: 0.114886	valid_1's auc: 0.836311	valid_1's binary_logloss: 0.140581
    [46]	valid_0's auc: 0.898034	valid_0's binary_logloss: 0.114544	valid_1's auc: 0.835871	valid_1's binary_logloss: 0.140663
    [47]	valid_0's auc: 0.898562	valid_0's binary_logloss: 0.114262	valid_1's auc: 0.835926	valid_1's binary_logloss: 0.140642
    [48]	valid_0's auc: 0.898919	valid_0's binary_logloss: 0.114006	valid_1's auc: 0.835849	valid_1's binary_logloss: 0.140687
    [49]	valid_0's auc: 0.899111	valid_0's binary_logloss: 0.113791	valid_1's auc: 0.835874	valid_1's binary_logloss: 0.140728
    [50]	valid_0's auc: 0.89987	valid_0's binary_logloss: 0.113543	valid_1's auc: 0.835915	valid_1's binary_logloss: 0.14075
    [51]	valid_0's auc: 0.90004	valid_0's binary_logloss: 0.113342	valid_1's auc: 0.835947	valid_1's binary_logloss: 0.140748
    [52]	valid_0's auc: 0.900405	valid_0's binary_logloss: 0.113087	valid_1's auc: 0.836011	valid_1's binary_logloss: 0.140767
    [53]	valid_0's auc: 0.900828	valid_0's binary_logloss: 0.112831	valid_1's auc: 0.836259	valid_1's binary_logloss: 0.140771
    [54]	valid_0's auc: 0.901597	valid_0's binary_logloss: 0.112604	valid_1's auc: 0.836296	valid_1's binary_logloss: 0.14078
    [55]	valid_0's auc: 0.901645	valid_0's binary_logloss: 0.112429	valid_1's auc: 0.836095	valid_1's binary_logloss: 0.140822
    [56]	valid_0's auc: 0.902162	valid_0's binary_logloss: 0.112169	valid_1's auc: 0.835965	valid_1's binary_logloss: 0.14086
    [57]	valid_0's auc: 0.902422	valid_0's binary_logloss: 0.111944	valid_1's auc: 0.835493	valid_1's binary_logloss: 0.140993
    Early stopping, best iteration is:
    [27]	valid_0's auc: 0.884676	valid_0's binary_logloss: 0.121001	valid_1's auc: 0.838046	valid_1's binary_logloss: 0.140086
    [1]	training's auc: 0.824305	training's binary_logloss: 0.156217	valid_1's auc: 0.819488	valid_1's binary_logloss: 0.165016
    Training until validation scores don't improve for 30 rounds
    [2]	training's auc: 0.828798	training's binary_logloss: 0.150959	valid_1's auc: 0.822075	valid_1's binary_logloss: 0.159734
    [3]	training's auc: 0.839609	training's binary_logloss: 0.147147	valid_1's auc: 0.829436	valid_1's binary_logloss: 0.156119
    [4]	training's auc: 0.845158	training's binary_logloss: 0.144107	valid_1's auc: 0.836147	valid_1's binary_logloss: 0.153073
    [5]	training's auc: 0.847711	training's binary_logloss: 0.14162	valid_1's auc: 0.839041	valid_1's binary_logloss: 0.150773
    [6]	training's auc: 0.849184	training's binary_logloss: 0.139622	valid_1's auc: 0.839076	valid_1's binary_logloss: 0.148948
    [7]	training's auc: 0.85094	training's binary_logloss: 0.13786	valid_1's auc: 0.839943	valid_1's binary_logloss: 0.147346
    [8]	training's auc: 0.853386	training's binary_logloss: 0.136418	valid_1's auc: 0.84098	valid_1's binary_logloss: 0.146068
    [9]	training's auc: 0.854751	training's binary_logloss: 0.135188	valid_1's auc: 0.840686	valid_1's binary_logloss: 0.14506
    [10]	training's auc: 0.855887	training's binary_logloss: 0.134098	valid_1's auc: 0.841299	valid_1's binary_logloss: 0.144134
    [11]	training's auc: 0.856935	training's binary_logloss: 0.133117	valid_1's auc: 0.841659	valid_1's binary_logloss: 0.14327
    [12]	training's auc: 0.858464	training's binary_logloss: 0.132253	valid_1's auc: 0.841543	valid_1's binary_logloss: 0.14261
    [13]	training's auc: 0.859951	training's binary_logloss: 0.131471	valid_1's auc: 0.841645	valid_1's binary_logloss: 0.14205
    [14]	training's auc: 0.861343	training's binary_logloss: 0.130767	valid_1's auc: 0.841389	valid_1's binary_logloss: 0.14164
    [15]	training's auc: 0.863266	training's binary_logloss: 0.130102	valid_1's auc: 0.84154	valid_1's binary_logloss: 0.141254
    [16]	training's auc: 0.864645	training's binary_logloss: 0.129469	valid_1's auc: 0.841108	valid_1's binary_logloss: 0.140999
    [17]	training's auc: 0.865605	training's binary_logloss: 0.128901	valid_1's auc: 0.840563	valid_1's binary_logloss: 0.140752
    [18]	training's auc: 0.866635	training's binary_logloss: 0.128334	valid_1's auc: 0.839571	valid_1's binary_logloss: 0.140569
    [19]	training's auc: 0.867769	training's binary_logloss: 0.127836	valid_1's auc: 0.839656	valid_1's binary_logloss: 0.14032
    [20]	training's auc: 0.868754	training's binary_logloss: 0.127334	valid_1's auc: 0.839451	valid_1's binary_logloss: 0.140153
    [21]	training's auc: 0.86983	training's binary_logloss: 0.12692	valid_1's auc: 0.839806	valid_1's binary_logloss: 0.139937
    [22]	training's auc: 0.870884	training's binary_logloss: 0.126484	valid_1's auc: 0.839529	valid_1's binary_logloss: 0.13983
    [23]	training's auc: 0.871649	training's binary_logloss: 0.126082	valid_1's auc: 0.839217	valid_1's binary_logloss: 0.139727
    [24]	training's auc: 0.872461	training's binary_logloss: 0.125727	valid_1's auc: 0.838771	valid_1's binary_logloss: 0.139684
    [25]	training's auc: 0.873292	training's binary_logloss: 0.125365	valid_1's auc: 0.838891	valid_1's binary_logloss: 0.139609
    [26]	training's auc: 0.874599	training's binary_logloss: 0.124992	valid_1's auc: 0.839175	valid_1's binary_logloss: 0.139492
    [27]	training's auc: 0.875485	training's binary_logloss: 0.124654	valid_1's auc: 0.83916	valid_1's binary_logloss: 0.139441
    [28]	training's auc: 0.876195	training's binary_logloss: 0.124346	valid_1's auc: 0.838877	valid_1's binary_logloss: 0.139445
    [29]	training's auc: 0.877178	training's binary_logloss: 0.124027	valid_1's auc: 0.839368	valid_1's binary_logloss: 0.139322
    [30]	training's auc: 0.878447	training's binary_logloss: 0.123667	valid_1's auc: 0.838922	valid_1's binary_logloss: 0.139324
    [31]	training's auc: 0.879197	training's binary_logloss: 0.123402	valid_1's auc: 0.838453	valid_1's binary_logloss: 0.139316
    [32]	training's auc: 0.880183	training's binary_logloss: 0.123092	valid_1's auc: 0.838572	valid_1's binary_logloss: 0.139283
    [33]	training's auc: 0.881377	training's binary_logloss: 0.122805	valid_1's auc: 0.838535	valid_1's binary_logloss: 0.139271
    [34]	training's auc: 0.882181	training's binary_logloss: 0.122567	valid_1's auc: 0.83825	valid_1's binary_logloss: 0.139275
    [35]	training's auc: 0.883237	training's binary_logloss: 0.122275	valid_1's auc: 0.838533	valid_1's binary_logloss: 0.139208
    [36]	training's auc: 0.884433	training's binary_logloss: 0.121989	valid_1's auc: 0.838446	valid_1's binary_logloss: 0.139217
    [37]	training's auc: 0.885423	training's binary_logloss: 0.121707	valid_1's auc: 0.838379	valid_1's binary_logloss: 0.139221
    [38]	training's auc: 0.88628	training's binary_logloss: 0.121411	valid_1's auc: 0.838156	valid_1's binary_logloss: 0.139254
    [39]	training's auc: 0.886985	training's binary_logloss: 0.121175	valid_1's auc: 0.838432	valid_1's binary_logloss: 0.139181
    [40]	training's auc: 0.887543	training's binary_logloss: 0.120933	valid_1's auc: 0.838247	valid_1's binary_logloss: 0.139215
    [41]	training's auc: 0.888425	training's binary_logloss: 0.120677	valid_1's auc: 0.83826	valid_1's binary_logloss: 0.139218
    Early stopping, best iteration is:
    [11]	training's auc: 0.856935	training's binary_logloss: 0.133117	valid_1's auc: 0.841659	valid_1's binary_logloss: 0.14327
    GridSearchCV 최적 파라미머: {'max_depth': 128, 'min_child_samples': 100, 'num_leaves': 32, 'subsample': 0.8}
    ROC AUC: 0.8417
    


```python

```
