import numpy as np

# total # of events
# before Pcharged and cos_thet cuts
ee_total = 93802
mm_total = 94381
tt_total = 79214
qq_total = 98563
#after Pcharged and cos_thet cuts
ee_total = 20499
mm_total = 89646
tt_total = 79099
qq_total = 98100

# cut for ee s-channel
ee_ee = 19817
ee_ee /= ee_total
ee_mm = 0
ee_mm /= mm_total
ee_tt = 651
ee_tt /= tt_total
ee_qq = 0
ee_qq /= tt_total
eps1 = np.array([ee_ee, ee_mm, ee_tt, ee_qq])

# cut for mm
mm_ee = 144
mm_ee /= ee_total
mm_mm = 83228
mm_mm /= mm_total
mm_tt = 480
mm_tt /= tt_total
mm_qq = 0
mm_qq /= qq_total
eps2 = np.array([mm_ee, mm_mm, mm_tt, mm_qq])

# cut for tt
tt_ee = 314
tt_ee /= ee_total
tt_mm = 2841
tt_mm /= mm_total
tt_tt = 70250
tt_tt /= tt_total
tt_qq = 39
tt_qq /= qq_total
eps3 = np.array([tt_ee, tt_mm, tt_tt, tt_qq])

# cut for qq
qq_ee = 0
qq_ee /= ee_total
qq_mm = 0
qq_mm /= mm_total
qq_tt = 78
qq_tt /= tt_total
qq_qq = 92688
qq_qq /= qq_total
eps4 = np.array([qq_ee, qq_mm, qq_tt, qq_qq])

eps = np.array([eps1, eps2, eps3, eps4])
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
print(eps)
