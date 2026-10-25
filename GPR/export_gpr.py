"""Processing provenance and numeric/figure exports."""
import json
from pathlib import Path
import numpy as np


def export_section(gpr, basename, history, geometry=None):
    """Export arrays and JSON history; keep original source files untouched."""
    name=Path(basename)
    np.savez_compressed(str(name)+'.npz',amplitudes=gpr.amplitudes,
                        time_ns=gpr.time_ns,distance_m=gpr.distance_m)
    info={'source_name':gpr.name,'steps':history,'units':{
        'time':'ns','distance':'m','amplitudes':'instrument units or processed relative amplitudes'},
        'geometry_csv':str(geometry) if geometry else None}
    Path(str(name)+'.json').write_text(json.dumps(info,indent=2))
    return str(name)+'.npz',str(name)+'.json'
