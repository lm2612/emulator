def CO2_MMR_ppm(MMR):
    return (28.9644 / 44.0095) * 1e6 * MMR

def CH4_MMR_ppb(MMR):
    return (28.9644 / 16.0425) * 1e9 * MMR

def CO2_ppm_MMR(ppm):
    return (44.0095 / 28.9644) * 1e-6 * ppm
    
def CH4_ppb_MMR(ppb):
    return (16.0425 / 28.9644) * 1e-9 * ppb
