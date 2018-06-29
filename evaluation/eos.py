from functools import lru_cache
from pathlib import Path

import menpo.io as mio
from menpo.shape import ColouredTriMesh

THIS_DIR = Path(__file__).parent



@lru_cache()
def load_eos_low_res_lm_index():
    return mio.import_pickle(
        THIS_DIR / 'eos_landmark_settings.pkl')['ibug_49_index']


@lru_cache()
def load_basel_kf_trilist():
    return mio.import_pickle(THIS_DIR / 'trilist.pkl')


@lru_cache()
def load_fw_on_eos_low_res_settings():
    d = mio.import_pickle(THIS_DIR / 'fw_on_eos_low_res_settings.pkl')
    bc_fw_on_eos_low_res = d['bc_fw_on_eos_low_res']
    tri_index_fw_on_eos_low_res = d['tri_index_fw_on_eos_low_res']
    return bc_fw_on_eos_low_res, tri_index_fw_on_eos_low_res


def upsample_eos_low_res_to_fw_no_texture(eos_mesh_low_res):
    bc_fw_on_eos_low_res, tri_index_fw_on_eos_low_res = load_fw_on_eos_low_res_settings()
    effective_fw_pc = eos_mesh_low_res.project_barycentric_coordinates(
        bc_fw_on_eos_low_res, tri_index_fw_on_eos_low_res)

    effective_fw = ColouredTriMesh(effective_fw_pc.points,
                                   trilist=load_basel_kf_trilist())
    return effective_fw
