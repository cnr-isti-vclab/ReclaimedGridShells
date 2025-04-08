import pymeshlab
import numpy as np
import os

def isotropic_remesh(input_mesh, target_lengths, base_path, output_prefix):
    for target_length in target_lengths:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(input_mesh)
        ms.meshing_isotropic_explicit_remeshing(iterations=300, targetlen=pymeshlab.PureValue(target_length), reprojectflag=False)
        output_mesh = f"{output_prefix}_{target_length:.1f}.ply"
        ms.save_current_mesh(os.path.join(base_path, output_mesh))
        print(f"Remeshing with target length {target_length:.1f} completed. Output saved as {output_mesh}")

if __name__ == "__main__":
    input_mesh = "C:\\Users\\Andrea\\Documents\\GitHub\\GeomDL5GridShell\\meshes\\Hall.ply"
    os.makedirs("C:\\Users\\Andrea\\Documents\\GitHub\\GeomDL5GridShell\meshes\\remeshing_cases", exist_ok=True)
    base_path = "C:\\Users\\Andrea\\Documents\\GitHub\\GeomDL5GridShell\\meshes\\remeshing_cases"
    target_lengths = np.linspace(0.7, 2.2, 15)
    output_prefix = "hall"  # Prefix for output file names
    isotropic_remesh(input_mesh, target_lengths, base_path, output_prefix)