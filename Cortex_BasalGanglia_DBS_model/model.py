import numpy as np
from pyNN import space
from pyNN.parameters import Sequence
from pyNN.neuron import (
    Population,
    StepCurrentSource,
    SpikeSourceArray,
    DCSource,
    Projection,
    StaticSynapse,
    FromFileConnector,
    NoisyCurrentSource,
    FixedNumberPreConnector,
)
from pyNN.random import RandomDistribution, NumpyRNG
from Cortical_Basal_Ganglia_Cell_Classes import (
    Cortical_Neuron_Type,
    Interneuron_Type,
    STN_Neuron_Type,
    GP_Neuron_Type,
    Thalamic_Neuron_Type,
)
from Electrode_Distances import (
    distances_to_electrode,
    collateral_distances_to_electrode,
)
import utils as u
from pathlib import Path


def create_network(
    Pop_size,
    steady_state_duration,
    sim_total_time,
    simulation_runtime,
    v_init,
    config,
):
    rng_seed = config.RandomSeed
    beta_burst_modulation_scale = config.beta_burst_modulation_scale
    ctx_dc_offset = config.ctx_dc_offset
    ctx_slow_modulation_amplitude = config.ctx_slow_modulation_amplitude
    ctx_slow_modulation_step_count = config.ctx_slow_modulation_step_count

    np.random.seed(rng_seed)
    structure_save_dir = Path("network_structure")

    # Sphere with radius 2000 um
    STN_space = space.RandomStructure(
        boundary=space.Sphere(2000), rng=NumpyRNG(seed=rng_seed)
    )

    # Generate Poisson-distributed Striatal Spike trains
    striatal_spike_times = u.generate_poisson_spike_times(
        Pop_size, steady_state_duration, simulation_runtime, 20, 1.0, rng_seed
    )

    # Save spike times so they can be reloaded
    np.save(structure_save_dir / "Striatal_Spike_Times.npy", striatal_spike_times)

    for i in range(Pop_size):
        spike_times = striatal_spike_times[i][0].value
        spike_times = spike_times[spike_times > steady_state_duration]
        striatal_spike_times[i][0] = Sequence(spike_times)

    # Generate the cortico-basal ganglia neuron populations
    Cortical_Pop = Population(
        Pop_size,
        Cortical_Neuron_Type(soma_bias_current_amp=0.245),
        structure=STN_space,
        label="Cortical Neurons",
    )
    Interneuron_Pop = Population(
        Pop_size,
        Interneuron_Type(bias_current_amp=0.070),
        initial_values={"v": v_init},
        label="Interneurons",
    )
    STN_Pop = Population(
        Pop_size,
        STN_Neuron_Type(bias_current=-0.125),
        structure=STN_space,
        initial_values={"v": v_init},
        label="STN Neurons",
    )
    # GPe/i have the same parameters, but different bias currents
    GPe_Pop = Population(
        Pop_size,
        GP_Neuron_Type(bias_current=-0.009),
        initial_values={"v": v_init},
        label="GPe Neurons",
    )
    GPi_Pop = Population(
        Pop_size,
        GP_Neuron_Type(bias_current=0.006),
        initial_values={"v": v_init},
        label="GPi Neurons",
    )
    Striatal_Pop = Population(
        Pop_size,
        SpikeSourceArray(spike_times=striatal_spike_times[0][0]),
        label="Striatal Neuron Spike Source",
    )
    Thalamic_Pop = Population(
        Pop_size,
        Thalamic_Neuron_Type(bias_current_density=0.00),
        initial_values={"v": v_init},
        label="Thalamic Neurons",
    )

    Striatal_Pop.set(spike_times=striatal_spike_times[:, 0])

    # Load burst times
    burst_dir = Path("burst_data")
    burst_times_script = burst_dir / "burst_times_1.txt"
    burst_level_script = burst_dir / "burst_level_1.txt"
    modulation_t = np.loadtxt(burst_times_script, delimiter=",")
    modulation_s = np.loadtxt(burst_level_script, delimiter=",")
    modulation_s = beta_burst_modulation_scale * modulation_s  # Scale the modulation signal
    cortical_modulation_current = StepCurrentSource(
        times=modulation_t, amplitudes=modulation_s
    )
    Cortical_Pop.inject(cortical_modulation_current)
    if ctx_dc_offset > 0:
        Cortical_Pop.inject(
            DCSource(start=steady_state_duration,
                     stop=sim_total_time,
                     amplitude=ctx_dc_offset))

    add_slow_modulation(
        Cortical_Pop,
        ctx_slow_modulation_amplitude,
        ctx_slow_modulation_step_count,
        steady_state_duration,
        sim_total_time
    )

    # Generate Noisy current sources for cortical pyramidal and interneuron populations
    # Inject each membrane noise current into each cortical and interneuron in network
    for cell in Cortical_Pop:
        cell.inject(
            NoisyCurrentSource(
                mean=0,
                stdev=0.005,
                start=steady_state_duration,
                stop=sim_total_time,
                dt=1.0,
            )
        )

    for cell in Interneuron_Pop:
        cell.inject(
            NoisyCurrentSource(
                mean=0,
                stdev=0.005,
                start=steady_state_duration,
                stop=sim_total_time,
                dt=1.0,
            )
        )

    # Position Check -
    # 1) Make sure cells are bounded in 4mm space in x, y coordinates
    # 2) Make sure no cells are placed inside the stimulating/recording
    # electrode -0.5mm<x<0.5mm, -1.5mm<y<2mm
    for Cortical_cell in Cortical_Pop:
        while (
            (np.abs(Cortical_cell.position[0]) > 2000)
            or ((np.abs(Cortical_cell.position[1]) > 2000))
        ) or (
            (np.abs(Cortical_cell.position[0]) < 500)
            and (-1500 < Cortical_cell.position[1] < 2000)
        ):
            Cortical_cell.position = STN_space.generate_positions(1).flatten()

    # Save the generated cortical xy positions to a textfile
    np.savetxt(structure_save_dir / "cortical_xy_pos.txt", Cortical_Pop.positions, delimiter=",")

    for STN_cell in STN_Pop:
        while (
            (np.abs(STN_cell.position[0]) > 2000)
            or ((np.abs(STN_cell.position[1]) > 2000))
        ) or (
            (np.abs(STN_cell.position[0]) < 500)
            and (-1500 < STN_cell.position[1] < 2000)
        ):
            STN_cell.position = STN_space.generate_positions(1).flatten()
        STN_cell.position[2] = 500

    # Save the generated STN xy positions to a textfile
    np.savetxt(structure_save_dir / "STN_xy_pos.txt", STN_Pop.positions, delimiter=",")

    # Synaptic Connections
    # Add variability to Cortical connections - cortical interneuron
    # connection weights are random from uniform distribution
    gCtxInt_max_weight = 2.5e-3  # Ctx -> Int max coupling value
    gIntCtx_max_weight = 6.0e-3  # Int -> Ctx max coupling value
    gCtxInt = RandomDistribution(
        "uniform", (0, gCtxInt_max_weight), rng=NumpyRNG(seed=rng_seed)
    )
    gIntCtx = RandomDistribution(
        "uniform", (0, gIntCtx_max_weight), rng=NumpyRNG(seed=rng_seed)
    )

    # Define other synaptic connection weights and delays
    syn_CorticalAxon_Interneuron = StaticSynapse(weight=gCtxInt, delay=2)
    syn_Interneuron_CorticalSoma = StaticSynapse(weight=gIntCtx, delay=2)
    # syn_CorticalSpikeSourceCorticalAxon = StaticSynapse(weight=0.25, delay=0)
    syn_CorticalCollateralSTN = StaticSynapse(weight=0.12, delay=1)
    syn_STNGPe = StaticSynapse(weight=0.111111, delay=4)
    syn_GPeGPe = StaticSynapse(weight=0.015, delay=4)
    syn_GPeSTN = StaticSynapse(weight=0.111111, delay=3)
    syn_StriatalGPe = StaticSynapse(weight=0.01, delay=1)
    syn_STNGPi = StaticSynapse(weight=0.111111, delay=2)
    syn_GPeGPi = StaticSynapse(weight=0.111111, delay=2)
    syn_GPiThalamic = StaticSynapse(weight=3.0, delay=2)
    syn_ThalamicCortical = StaticSynapse(weight=5, delay=2)
    syn_CorticalThalamic = StaticSynapse(weight=0.0, delay=2)

    # Create new network topology Connections
    prj_CorticalAxon_Interneuron = Projection(
        Cortical_Pop,
        Interneuron_Pop,
        FixedNumberPreConnector(n=10, allow_self_connections=False),
        syn_CorticalAxon_Interneuron,
        source="middle_axon_node",
        receptor_type="AMPA",
    )
    prj_Interneuron_CorticalSoma = Projection(
        Interneuron_Pop,
        Cortical_Pop,
        FixedNumberPreConnector(n=10, allow_self_connections=False),
        syn_Interneuron_CorticalSoma,
        receptor_type="GABAa",
    )
    prj_CorticalSTN = Projection(
        Cortical_Pop,
        STN_Pop,
        FixedNumberPreConnector(n=5, allow_self_connections=False),
        syn_CorticalCollateralSTN,
        source="collateral(0.5)",
        receptor_type="AMPA",
    )
    prj_STNGPe = Projection(
        STN_Pop,
        GPe_Pop,
        FixedNumberPreConnector(n=1, allow_self_connections=False),
        syn_STNGPe,
        source="soma(0.5)",
        receptor_type="AMPA",
    )
    prj_GPeGPe = Projection(
        GPe_Pop,
        GPe_Pop,
        FixedNumberPreConnector(n=1, allow_self_connections=False),
        syn_GPeGPe,
        source="soma(0.5)",
        receptor_type="GABAa",
    )
    prj_GPeSTN = Projection(
        GPe_Pop,
        STN_Pop,
        FixedNumberPreConnector(n=2, allow_self_connections=False),
        syn_GPeSTN,
        source="soma(0.5)",
        receptor_type="GABAa",
    )
    prj_StriatalGPe = Projection(
        Striatal_Pop,
        GPe_Pop,
        FixedNumberPreConnector(n=1, allow_self_connections=False),
        syn_StriatalGPe,
        source="soma(0.5)",
        receptor_type="GABAa",
    )
    prj_STNGPi = Projection(
        STN_Pop,
        GPi_Pop,
        FixedNumberPreConnector(n=1, allow_self_connections=False),
        syn_STNGPi,
        source="soma(0.5)",
        receptor_type="AMPA",
    )
    prj_GPeGPi = Projection(
        GPe_Pop,
        GPi_Pop,
        FixedNumberPreConnector(n=1, allow_self_connections=False),
        syn_GPeGPi,
        source="soma(0.5)",
        receptor_type="GABAa",
    )
    prj_GPiThalamic = Projection(
        GPi_Pop,
        Thalamic_Pop,
        FixedNumberPreConnector(n=1, allow_self_connections=False),
        syn_GPiThalamic,
        source="soma(0.5)",
        receptor_type="GABAa",
    )
    prj_ThalamicCortical = Projection(
        Thalamic_Pop,
        Cortical_Pop,
        FixedNumberPreConnector(n=1, allow_self_connections=False),
        syn_ThalamicCortical,
        source="soma(0.5)",
        receptor_type="AMPA",
    )
    prj_CorticalThalamic = Projection(
        Cortical_Pop,
        Thalamic_Pop,
        FixedNumberPreConnector(n=1, allow_self_connections=False),
        syn_CorticalThalamic,
        source="soma(0.5)",
        receptor_type="AMPA",
    )

    # Save the network topology so it can be reloaded
    # prj_CorticalSpikeSourceCorticalSoma.saveConnections(file="CorticalSpikeSourceCorticalSoma_Connections.txt")
    prj_CorticalAxon_Interneuron.saveConnections(
        file=structure_save_dir / "CorticalAxonInterneuron_Connections.txt"
    )
    prj_Interneuron_CorticalSoma.saveConnections(
        file=structure_save_dir / "InterneuronCortical_Connections.txt"
    )
    prj_CorticalSTN.saveConnections(file=structure_save_dir / "CorticalSTN_Connections.txt")
    prj_STNGPe.saveConnections(file=structure_save_dir / "STNGPe_Connections.txt")
    prj_GPeGPe.saveConnections(file=structure_save_dir / "GPeGPe_Connections.txt")
    prj_GPeSTN.saveConnections(file=structure_save_dir / "GPeSTN_Connections.txt")
    prj_StriatalGPe.saveConnections(file=structure_save_dir / "StriatalGPe_Connections.txt")
    prj_STNGPi.saveConnections(file=structure_save_dir / "STNGPi_Connections.txt")
    prj_GPeGPi.saveConnections(file=structure_save_dir / "GPeGPi_Connections.txt")
    prj_GPiThalamic.saveConnections(file=structure_save_dir / "GPiThalamic_Connections.txt")
    prj_ThalamicCortical.saveConnections(file=structure_save_dir / "ThalamicCorticalSoma_Connections.txt")
    prj_CorticalThalamic.saveConnections(file=structure_save_dir / "CorticalSomaThalamic_Connections.txt")
    # Load GPe stimulation order
    GPe_stimulation_order = np.loadtxt(structure_save_dir / "GPe_Stimulation_Order.txt", delimiter=",")
    GPe_stimulation_order = [int(index) for index in GPe_stimulation_order]

    return (
        striatal_spike_times,
        Cortical_Pop,
        Interneuron_Pop,
        STN_Pop,
        GPe_Pop,
        GPi_Pop,
        Striatal_Pop,
        Thalamic_Pop,
        prj_CorticalAxon_Interneuron,
        prj_Interneuron_CorticalSoma,
        prj_CorticalSTN,
        prj_STNGPe,
        prj_GPeGPe,
        prj_GPeSTN,
        prj_StriatalGPe,
        prj_STNGPi,
        prj_GPeGPi,
        prj_GPiThalamic,
        prj_ThalamicCortical,
        prj_CorticalThalamic,
        GPe_stimulation_order,
    )


def load_network(
    steady_state_duration,
    sim_total_time,
    simulation_runtime,
    v_init,
    config,
):
    rng_seed = config.RandomSeed
    beta_burst_modulation_scale = config.beta_burst_modulation_scale
    ctx_dc_offset = config.ctx_dc_offset
    ctx_slow_modulation_amplitude = config.ctx_slow_modulation_amplitude
    ctx_slow_modulation_step_count = config.ctx_slow_modulation_step_count

    structure_save_dir = Path("network_structure")

    if config.cortical_beta_mechanism == "spikes":
        ctx_beta_spike_input = True
        ctx_beta_frequency = config.ctx_beta_spike_frequency
        ctx_beta_isi_dither = config.ctx_beta_spike_isi_dither
        ctx_beta_synapse_strength = config.ctx_beta_synapse_strength
    elif config.cortical_beta_mechanism == "current":
        ctx_beta_spike_input = False
    else:
        print(f"I don't know how to modulate cortical beta using {config.cortical_beta_mechanism}")

    np.random.seed(rng_seed)
    # Sphere with radius 2000 um
    STN_space = space.RandomStructure(
        boundary=space.Sphere(2000), rng=NumpyRNG(seed=rng_seed)
    )

    # Load striatal spike times from file
    striatal_spike_times = np.load(structure_save_dir / "Striatal_Spike_Times.npy", allow_pickle=True)
    Pop_size = len(striatal_spike_times[:, 0])
    for i in range(Pop_size):
        spike_times = striatal_spike_times[i][0].value
        spike_times = spike_times[spike_times > steady_state_duration]
        striatal_spike_times[i][0] = Sequence(spike_times)

    # Generate the cortico-basal ganglia neuron populations
    Cortical_Pop = Population(
        Pop_size,
        Cortical_Neuron_Type(soma_bias_current_amp=0.245),
        structure=STN_space,
        label="Cortical Neurons",
    )
    Interneuron_Pop = Population(
        Pop_size,
        Interneuron_Type(bias_current_amp=0.070),
        initial_values={"v": v_init},
        label="Interneurons",
    )
    STN_Pop = Population(
        Pop_size,
        STN_Neuron_Type(bias_current=-0.125),
        structure=STN_space,
        initial_values={"v": v_init},
        label="STN Neurons",
    )
    # GPe/i have the same parameters, but different bias currents
    GPe_Pop = Population(
        Pop_size,
        GP_Neuron_Type(bias_current=-0.009),
        initial_values={"v": v_init},
        label="GPe Neurons",
    )
    GPi_Pop = Population(
        Pop_size,
        GP_Neuron_Type(bias_current=0.006),
        initial_values={"v": v_init},
        label="GPi Neurons",
    )
    Striatal_Pop = Population(
        Pop_size,
        SpikeSourceArray(spike_times=striatal_spike_times[0][0]),
        label="Striatal Neuron Spike Source",
    )
    Thalamic_Pop = Population(
        Pop_size,
        Thalamic_Neuron_Type(),
        initial_values={"v": v_init},
        label="Thalamic Neurons",
    )

    Striatal_Pop.set(spike_times=striatal_spike_times[:, 0])

    # Generate Noisy current sources for cortical pyramidal and interneuron populations
    # Inject each membrane noise current into each cortical and interneuron in network
    for cell in Cortical_Pop:
        cell.inject(
            NoisyCurrentSource(
                mean=0,
                stdev=0.005,
                start=steady_state_duration,
                stop=sim_total_time,
                dt=1.0,
            )
        )

    for cell in Interneuron_Pop:
        cell.inject(
            NoisyCurrentSource(
                mean=0,
                stdev=0.005,
                start=steady_state_duration,
                stop=sim_total_time,
                dt=1.0,
            )
        )

    # Load burst times
    burst_dir = Path("burst_data")
    burst_times_script = burst_dir / "burst_times_1.txt"
    burst_level_script = burst_dir / "burst_level_1.txt"
    modulation_t = np.loadtxt(burst_times_script, delimiter=",")
    modulation_s = np.loadtxt(burst_level_script, delimiter=",")

    while modulation_t[-1] < sim_total_time:
        time_shift = int(modulation_t[-1] - modulation_t[0] + np.mean(np.diff(modulation_t)))
        modulation_t = np.hstack((modulation_t, time_shift + modulation_t))
        modulation_s = np.hstack((modulation_s, modulation_s))

    if modulation_t[0] > sim_total_time:
        time_shift = int(steady_state_duration - modulation_t[0])
        modulation_t += time_shift

    random_array = np.random.randn(Pop_size)

    for cell in Cortical_Pop:
        cell_offset = ctx_dc_offset
        if config.ctx_dc_offset_std > 0:
            cell_offset += config.ctx_dc_offset_std * random_array[cell]
        cell.inject(
            DCSource(
                start=steady_state_duration,
                stop=sim_total_time,
                amplitude=cell_offset))

    if ctx_beta_spike_input:
        ctx_poisson_tt, ctx_poisson_a = u.burst_txt_to_signal(modulation_t, ctx_beta_frequency * (modulation_s + 1), steady_state_duration, sim_total_time, dt=0.1)
        ctx_spike_times = u.generate_inhomogeneous_poisson_spike_times(
            Pop_size,
            ctx_poisson_tt,
            ctx_poisson_a,
            dt=0.1,
            random_seed=rng_seed,
            isi_dither=ctx_beta_isi_dither,
        )
        Ctx_Beta_Source_Pop = Population(
            Pop_size,
            SpikeSourceArray(spike_times=ctx_spike_times[0]),
            label="Ctx Beta Spike Source",
            )
        syn_Ctx_Beta = StaticSynapse(weight=ctx_beta_synapse_strength, delay=1)
        prj_Ctx_Beta = Projection(
            Ctx_Beta_Source_Pop,
            Cortical_Pop,
            FixedNumberPreConnector(n=1, allow_self_connections=False),
            syn_Ctx_Beta,
            source="soma(0.5)",
            receptor_type="AMPA",
        )
    else:
        modulation_s = beta_burst_modulation_scale * modulation_s  # Scale the modulation signal

        cortical_burst_modulation_current = StepCurrentSource(
            times=modulation_t, amplitudes=modulation_s
        )
        Cortical_Pop.inject(cortical_burst_modulation_current)

        add_slow_modulation(
            Cortical_Pop,
            ctx_slow_modulation_amplitude,
            ctx_slow_modulation_step_count,
            steady_state_duration,
            sim_total_time
        )

    # Load cortical positions - Comment/Remove to generate new positions
    Cortical_Neuron_xy_Positions = np.loadtxt(structure_save_dir / "cortical_xy_pos.txt", delimiter=",")
    cortex_local_indices = [cell in Cortical_Pop for cell in Cortical_Pop.all_cells]
    Cortical_Neuron_x_Positions = Cortical_Neuron_xy_Positions[0, cortex_local_indices]
    Cortical_Neuron_y_Positions = Cortical_Neuron_xy_Positions[1, cortex_local_indices]

    # Set cortical xy positions to those loaded in
    for ii, cell in enumerate(Cortical_Pop):
        cell.position[0] = Cortical_Neuron_x_Positions[ii]
        cell.position[1] = Cortical_Neuron_y_Positions[ii]

    # Load STN positions - Comment/Remove to generate new positions
    STN_Neuron_xy_Positions = np.loadtxt(structure_save_dir / "STN_xy_pos.txt", delimiter=",")
    stn_local_indices = [cell in STN_Pop for cell in STN_Pop.all_cells]
    STN_Neuron_x_Positions = STN_Neuron_xy_Positions[0, stn_local_indices]
    STN_Neuron_y_Positions = STN_Neuron_xy_Positions[1, stn_local_indices]

    # Set STN xy positions to those loaded in
    for ii, cell in enumerate(STN_Pop):
        cell.position[0] = STN_Neuron_x_Positions[ii]
        cell.position[1] = STN_Neuron_y_Positions[ii]
        cell.position[2] = 500

    # Synaptic Connections
    # Add variability to Cortical connections - cortical interneuron
    # connection weights are random from uniform distribution
    gCtxInt_max_weight = 2.5e-3  # Ctx -> Int max coupling value
    gIntCtx_max_weight = 6.0e-3  # Int -> Ctx max coupling value
    gCtxInt = RandomDistribution(
        "uniform", (0, gCtxInt_max_weight), rng=NumpyRNG(seed=rng_seed)
    )
    gIntCtx = RandomDistribution(
        "uniform", (0, gIntCtx_max_weight), rng=NumpyRNG(seed=rng_seed)
    )

    # Define other synaptic connection weights and delays
    syn_CorticalAxon_Interneuron = StaticSynapse(weight=gCtxInt, delay=2)
    syn_Interneuron_CorticalSoma = StaticSynapse(weight=gIntCtx, delay=2)
    # syn_CorticalSpikeSourceCorticalAxon = StaticSynapse(weight=0.25, delay=0)
    syn_CorticalCollateralSTN = StaticSynapse(weight=0.12, delay=1)
    syn_STNGPe = StaticSynapse(weight=0.111111, delay=4)
    syn_GPeGPe = StaticSynapse(weight=0.015, delay=4)
    syn_GPeSTN = StaticSynapse(weight=0.111111, delay=3)
    syn_StriatalGPe = StaticSynapse(weight=0.01, delay=1)
    syn_STNGPi = StaticSynapse(weight=0.111111, delay=2)
    syn_GPeGPi = StaticSynapse(weight=0.111111, delay=2)
    syn_GPiThalamic = StaticSynapse(weight=3.0, delay=2)
    syn_ThalamicCortical = StaticSynapse(weight=5, delay=2)
    syn_CorticalThalamic = StaticSynapse(weight=0.0, delay=2)

    # Load network topology from file
    prj_CorticalAxon_Interneuron = Projection(
        Cortical_Pop,
        Interneuron_Pop,
        FromFileConnector(str(structure_save_dir / "CorticalAxonInterneuron_Connections.txt")),
        syn_CorticalAxon_Interneuron,
        source="middle_axon_node",
        receptor_type="AMPA",
    )
    prj_Interneuron_CorticalSoma = Projection(
        Interneuron_Pop,
        Cortical_Pop,
        FromFileConnector(str(structure_save_dir / "InterneuronCortical_Connections.txt")),
        syn_Interneuron_CorticalSoma,
        receptor_type="GABAa",
    )
    prj_CorticalSTN = Projection(
        Cortical_Pop,
        STN_Pop,
        FromFileConnector(str(structure_save_dir / "CorticalSTN_Connections.txt")),
        syn_CorticalCollateralSTN,
        source="collateral(0.5)",
        receptor_type="AMPA",
    )
    prj_STNGPe = Projection(
        STN_Pop,
        GPe_Pop,
        FromFileConnector(str(structure_save_dir / "STNGPe_Connections.txt")),
        syn_STNGPe,
        source="soma(0.5)",
        receptor_type="AMPA",
    )
    prj_GPeGPe = Projection(
        GPe_Pop,
        GPe_Pop,
        FromFileConnector(str(structure_save_dir / "GPeGPe_Connections.txt")),
        syn_GPeGPe,
        source="soma(0.5)",
        receptor_type="GABAa",
    )
    prj_GPeSTN = Projection(
        GPe_Pop,
        STN_Pop,
        FromFileConnector(str(structure_save_dir / "GPeSTN_Connections.txt")),
        syn_GPeSTN,
        source="soma(0.5)",
        receptor_type="GABAa",
    )
    prj_StriatalGPe = Projection(
        Striatal_Pop,
        GPe_Pop,
        FromFileConnector(str(structure_save_dir / "StriatalGPe_Connections.txt")),
        syn_StriatalGPe,
        source="soma(0.5)",
        receptor_type="GABAa",
    )
    prj_STNGPi = Projection(
        STN_Pop,
        GPi_Pop,
        FromFileConnector(str(structure_save_dir / "STNGPi_Connections.txt")),
        syn_STNGPi,
        source="soma(0.5)",
        receptor_type="AMPA",
    )
    prj_GPeGPi = Projection(
        GPe_Pop,
        GPi_Pop,
        FromFileConnector(str(structure_save_dir / "GPeGPi_Connections.txt")),
        syn_GPeGPi,
        source="soma(0.5)",
        receptor_type="GABAa",
    )
    prj_GPiThalamic = Projection(
        GPi_Pop,
        Thalamic_Pop,
        FromFileConnector(str(structure_save_dir / "GPiThalamic_Connections.txt")),
        syn_GPiThalamic,
        source="soma(0.5)",
        receptor_type="GABAa",
    )
    prj_ThalamicCortical = Projection(
        Thalamic_Pop,
        Cortical_Pop,
        FromFileConnector(str(structure_save_dir / "ThalamicCorticalSoma_Connections.txt")),
        syn_ThalamicCortical,
        source="soma(0.5)",
        receptor_type="AMPA",
    )
    prj_CorticalThalamic = Projection(
        Cortical_Pop,
        Thalamic_Pop,
        FromFileConnector(str(structure_save_dir / "CorticalSomaThalamic_Connections.txt")),
        syn_CorticalThalamic,
        source="soma(0.5)",
        receptor_type="AMPA",
    )

    # Load GPe stimulation order
    GPe_stimulation_order = np.loadtxt(structure_save_dir / "GPe_Stimulation_Order.txt", delimiter=",")
    GPe_stimulation_order = [int(index) for index in GPe_stimulation_order]

    return (
        Pop_size,
        striatal_spike_times,
        Cortical_Pop,
        Interneuron_Pop,
        STN_Pop,
        GPe_Pop,
        GPi_Pop,
        Striatal_Pop,
        Thalamic_Pop,
        prj_CorticalAxon_Interneuron,
        prj_Interneuron_CorticalSoma,
        prj_CorticalSTN,
        prj_STNGPe,
        prj_GPeGPe,
        prj_GPeSTN,
        prj_StriatalGPe,
        prj_STNGPi,
        prj_GPeGPi,
        prj_GPiThalamic,
        prj_ThalamicCortical,
        prj_CorticalThalamic,
        GPe_stimulation_order,
    )


def electrode_distance(
    recording_electrode_1_position,
    recording_electrode_2_position,
    STN_Pop,
    stimulating_electrode_position,
    Cortical_Pop,
):
    # Calculate STN cell distances to each recording electrode
    # using only xy coordinates for distance calculations
    STN_recording_electrode_1_distances = distances_to_electrode(
        recording_electrode_1_position, STN_Pop
    )
    STN_recording_electrode_2_distances = distances_to_electrode(
        recording_electrode_2_position, STN_Pop
    )

    # Calculate Cortical Collateral distances from the stimulating electrode -
    # using xyz coordinates for distance
    # calculation - these distances need to be in um for xtra mechanism
    Cortical_Collateral_stimulating_electrode_distances = (
        collateral_distances_to_electrode(
            stimulating_electrode_position, Cortical_Pop, L=500, nseg=11
        )
    )

    return (
        STN_recording_electrode_1_distances,
        STN_recording_electrode_2_distances,
        Cortical_Collateral_stimulating_electrode_distances,
    )


def add_slow_modulation(Population, amplitude, step_count, steady_state_duration, sim_total_time):
    if abs(amplitude) > 0:
        slow_modulation_start = steady_state_duration
        slow_modulation_stage_duration = (
            (sim_total_time - slow_modulation_start) / (step_count + 1)
            )
        for i in range(step_count + 1):
            stage_start = slow_modulation_start + i * slow_modulation_stage_duration
            stage_end = stage_start + slow_modulation_stage_duration
            stage_amplitude = amplitude * ((i) % 2)
            if stage_amplitude == 0:
                continue
            Population.inject(DCSource(
                start=stage_start,
                stop=stage_end,
                amplitude=stage_amplitude
            ))
