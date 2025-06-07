import numpy as np
from neo import NeoMatlabIO
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy.io

save_path = r"C:\Users\Seda\Documents\My Model\CBG_Model_Fleming\Seda_OpenLoop_Noise_NoStim_Results\CORTICAL_LFP.mat"

def compute_distances_to_electrode(
        electrode_x: float,
        electrode_y: float,
        population_x: np.ndarray,
        population_y: np.ndarray,
        minimum_distance: float = 0,
        ):
    '''Compute distances from the population cells to the electrode'''
    distances = np.sqrt(
        (population_x - electrode_x) ** 2 +
        (population_y - electrode_y) ** 2
        )
    distances[distances < minimum_distance] = minimum_distance
    return distances


def compute_cortical_lfp(
        ctx_lfp_dir: Path,
        interneuron_lfp_dir: Path,
        electrode_y: list,
        ctx_positions: tuple[np.ndarray, np.ndarray],
        int_positions: tuple[np.ndarray, np.ndarray],
        excluded_radius: float = 0.06,
        sigma: float = 0.27,
        ):
    gaba_data_file_ctx = ctx_lfp_dir / "Ctx_GABAa_i.mat"
    ampa_data_file_ctx = ctx_lfp_dir / "Ctx_AMPA_i.mat"
    gaba_data_file_int = interneuron_lfp_dir / "Interneuron_GABAa_i.mat"
    ampa_data_file_int = interneuron_lfp_dir / "Interneuron_AMPA_i.mat"
    gaba_data_ctx = NeoMatlabIO(gaba_data_file_ctx).read()
    ampa_data_ctx = NeoMatlabIO(ampa_data_file_ctx).read()
    gaba_data_int = NeoMatlabIO(gaba_data_file_int).read()
    ampa_data_int = NeoMatlabIO(ampa_data_file_int).read()

    tt = gaba_data_ctx[0].segments[0].analogsignals[0].times

    x_ctx, y_ctx = ctx_positions
    x_int, y_int = int_positions
    all_currents_ctx = (
        gaba_data_ctx[0].segments[0].analogsignals[0].as_array() + 
        ampa_data_ctx[0].segments[0].analogsignals[0].as_array()
        )
    all_currents_int = (
        gaba_data_int[0].segments[0].analogsignals[0].as_array() +
        ampa_data_int[0].segments[0].analogsignals[0].as_array()
    )

    # Units:
    # all_currents: nA
    # sigma: S / m
    # distances: mm
    # ---
    # lfp: microvolt

    electrode_lfp = []
    for y_e in electrode_y:
        distances_ctx_e = compute_distances_to_electrode(
            0, y_e, x_ctx, y_ctx, excluded_radius
            )
        distances_int_e = compute_distances_to_electrode(
            0, y_e, x_int, y_int, excluded_radius
            )
        lfp_ctx = np.sum(all_currents_ctx / distances_ctx_e / (4 * np.pi * sigma), axis=1)
        lfp_int = np.sum(all_currents_int / distances_int_e / (4 * np.pi * sigma), axis=1)
        lfp_both = lfp_ctx + lfp_int
        electrode_lfp.append(lfp_both)
    return tt, electrode_lfp


def generate_random_cell_xy_positions(
        low: float,
        high: float,
        count: int,
        ) -> tuple[np.ndarray, np.ndarray]:
    '''Generates random distribution of points in a circle'''
    distances = np.random.uniform(low=low, high=high, size=(count,))
    angles = np.random.uniform(0, 2 * np.pi, size=(count,))
    x = distances * np.cos(angles)
    y = distances * np.sin(angles)
    return (x, y)


def generate_electrode_positions(
        electrode_count: int,
        electrode_distance: float,
        ) -> np.ndarray:
    '''Generate positions of linearly evenly spaced electrodes along the y axis'''
    if electrode_count % 2 == 1:
        # Odd number of electrodes, so the first one is at zero
        electrode_y = [0]
        for i in range(electrode_count - 1):
            electrode_y.append(
                ((1 + np.floor(i / 2)) * (-1) ** i) * electrode_distance
                )
    else:
        electrode_y = [electrode_distance / 2, -electrode_distance / 2]
        for i in range(electrode_count - 2):
            electrode_y.append(
                (-1) ** i * (np.floor(i / 2) + 1 + 0.5) * electrode_distance
                )
    electrode_y = sorted(electrode_y)

    return electrode_y


def plot_ctx_cell_and_electrode_location(
        position_ctx: tuple[np.ndarray, np.ndarray],
        position_int: tuple[np.ndarray, np.ndarray],
        position_electrode: tuple[np.ndarray, np.ndarray],
        electrode_radius: float,
        max_radius: float,
        ) -> None:
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_aspect(1)
    x_ctx, y_ctx = position_ctx
    x_int, y_int = position_int
    x_electrode, y_electrode = position_electrode
    plot_ctx = ax.scatter(x_ctx, y_ctx, s=4, color="#1446A0")
    plot_int = ax.scatter(x_int, y_int, s=4, color="#929982")
    circle = plt.Circle((0, 0), max_radius, fill=False, color="black")
    for i, y in enumerate(y_electrode):
        x = x_electrode[i]
        plot_exc = plt.Circle((x, y), electrode_radius, fill=True, color="#F05860")
        ax.add_artist(plot_exc)
        ax.text(0 - electrode_radius / 2, y - electrode_radius / 2, str(i))
    ax.add_artist(circle)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"Location of cortex neurons around the electrode{'s' if len(y_electrode) > 1 else ''}")
    ax.legend([plot_ctx, plot_int, plot_exc], ["Pyramidal", "Interneuron", "Electrode and excluded zone"])


if __name__ == "__main__":
    open_dir = Path(r"C:\Users\Seda\Documents\My Model\CBG_Model_Fleming\Seda_OpenLoop_Noise_NoStim_Results")
    ctx_dir = open_dir / "Cortical_Pop"
    interneuron_dir = open_dir / "Interneuron_Pop"

    electrode_count = 3
    electrode_distance = 0.4
    excluded_radius = 0.06
    min_radius = 0.01
    max_radius = 1
    
    sigma = 0.27


    position_ctx = generate_random_cell_xy_positions(min_radius, max_radius, 100)
    position_int = generate_random_cell_xy_positions(min_radius, max_radius, 100)
    
    electrode_y = generate_electrode_positions(electrode_count, electrode_distance)

    plot_ctx_cell_and_electrode_location(
        position_ctx,
        position_int,
        ([0 for _ in electrode_y], electrode_y),
        excluded_radius,
        max_radius
    )

    tt, electrode_lfp_open = compute_cortical_lfp(
        ctx_dir,
        interneuron_dir,
        electrode_y,
        position_ctx,
        position_int,
        excluded_radius,
        sigma,
        )

    plt.figure()
    plt.plot(tt, electrode_lfp_open[0])
    plt.plot(tt, electrode_lfp_open[1])
    plt.plot(tt, electrode_lfp_open[2])

    plt.xlabel("Time [ms]")
    plt.ylabel("Cortical LFP [mV]")
    plt.xlim([1000, 2000])
    plt.show()

scipy.io.savemat(save_path, {
    'tt': tt,  
    'electrode_lfp_open': electrode_lfp_open  
})

print(f"Cortical LFP data saved to: {save_path}")
