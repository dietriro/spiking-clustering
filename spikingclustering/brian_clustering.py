from brian2 import *
from prettytable import PrettyTable


def plot_weights(w):
    rng = range(1, len(w[0, 0]) + 1)

    f, axes = subplots(n_inputs, n_rbf, sharey=True)
    f.subplots_adjust(hspace=0.0, wspace=0.0)

    for i_in in range(n_inputs):
        for i_rbf in range(n_rbf):
            weights = w[i_in, i_rbf]
            axes[i_in, i_rbf].bar(rng, weights)
            axes[i_in, i_rbf].plot([0, n_delays + 1], [w_max, w_max], color='r', linestyle='--', linewidth=0.5)
            axes[i_in, i_rbf].set_ylim([0, w_max + w_max * 0.15])

            if i_rbf == 0:
                axes[i_in, i_rbf].set_ylabel(f'Weights W - neurons_in-{i_in}')
            else:
                setp(axes[i_in, i_rbf].get_yticklabels(), visible=False)

            if i_in == 0:
                axes[i_in, i_rbf].set_title(f'Weights for neurons_rbf Neuron {i_rbf}')
                setp(axes[i_in, i_rbf].get_xticklabels(), visible=False)
            # ax1.set_xticks(range(0, len(weights_1)))

    show()


def plot_potential(m_pot, m_spikes):
    grid = GridSpec(4, 1)

    ax0 = subplot(grid[:3, 0])
    setp(ax0.get_xticklabels(), visible=False)
    for i in range(n_rbf):
        plot(m_pot.t / ms, m_pot.v[i] / mV, label=f'neurons_rbf[{i}]')
    plot([0, len(m_pot.t) * defaultclock.dt / 0.001], [Vt / ms, Vt / ms], color='r', linestyle='--', linewidth=1)
    ylabel('Membrane potential (in mV)')
    title('Membrane potential/spikes for neurons_rbf neurons')
    legend()

    subplot(grid[3, 0], sharex=ax0)
    plot(m_spikes.t / ms, m_spikes.i, '.')
    ylabel('neurons_rbf Neuron ID')
    xlabel('Time (in ms)')
    yticks(range(n_rbf))
    ylim([-0.5, 1.5])
    show()


def print_debugging_data():
    print('ti: ' + str(syn_stdp.ti[0, 0]))
    print('tj: ' + str(syn_stdp.tj[0, 0]))
    print('t_delta: ' + str(syn_stdp.delta_t[0, 0]))
    # print('t_delta2: ' + str(syn_stdp.delta_t2[0, 0]))
    # print('temp: ' + str(syn_stdp.temp[0, 0]))
    # print('temp2: ' + str(syn_stdp.temp2[0, 0]))
    print(syn_stdp.w_delta[0, 0])
    print(syn_stdp.w[0, 0])
    print(syn_stdp.w_new[0, 0])
    print()
    print('ti: ' + str(syn_stdp.ti[0, 1]))
    print('tj: ' + str(syn_stdp.tj[0, 1]))
    print('t_delta: ' + str(syn_stdp.delta_t[0, 1]))
    # print('t_delta2: ' + str(syn_stdp.delta_t2[0, 1]))
    # print('temp: ' + str(syn_stdp.temp[0, 1]))
    # print('temp2: ' + str(syn_stdp.temp2[0, 1]))
    print(syn_stdp.w_delta[0, 1])
    print(syn_stdp.w[0, 1])
    print(syn_stdp.w_new[0, 1])
    print()
    print()


## Constants
n_inputs = 2  # Number of input neurons/dimensions
n_rbf = 2  # Number of output/neurons_rbf neurons
n_delays = 16  # Number of delays per input-neurons_rbf connection

tau = 3.0 * ms  # Time constant tau for LiF model
w_max = 1 / (n_inputs * 4)  # Maximum weight of one synaptic connection
# w_max = 1 / (n_inputs * 3.2)      # Natschlaeger
# w_max = 1 / (n_inputs * 4.3)      # Bothe
w_factor = 1.0  # Weight factor for defining the maximum initialization weight relative to w_max
w_inh = -w_max * 3  # Inhibitory weight
d_back = -3.0  # Delay of the postsynaptic spike

# Natschlaeger
# b = -0.11
# c = -2.00
# beta = 1.11
# Bothe
# b = -0.3
# c = -2.85
# beta = 0.9
# Custom
b = -0.3  # Defines the max/min value of the learning function
c = -2.85  # Defines the distance to 0 of the learning function
beta = 0.9  # Defines the width of the learning function

lr = 0.01  # Learning-rate
Vt = 0.25  # Voltage threshold

defaultclock.dt = 0.1 * ms  # Simulation time-step
runtime = 50 * ms  # Runtime for one sample
runs = 50  # Number of total simulated runs
second_cluster = True  # Whether to use one or two clusters
n_clusters = second_cluster + 1  # The actual number of clusters

## Input

# Parameters
rand_clusters = True  # Whether to pick samples from random clusters or not
sigma = 0.5  # Standard deviation of the clusters
if n_inputs == 1:
    centers = [[2], [6]]  # Multi-dimensional array containing cluster centers (cluster_id, input_dim)
elif n_inputs == 2:
    centers = [[1, 2], [2, 9]]  # Multi-dimensional array containing cluster centers (cluster_id, input_dim)
else:
    centers = [[2, 2, 2], [2, 6, 7]]  # Multi-dimensional array containing cluster centers (cluster_id, input_dim)

# Generate input data from defined clusters
indices = array(range(n_inputs))
inputs = zeros([second_cluster + 1, runs, n_inputs]) * ms

for c in range(second_cluster + 1):
    for i in range(n_inputs):
        inputs[c, :, i] = np.random.normal(centers[c][i], sigma, runs).astype(int) * ms

## Code

# Create input neurons
neurons_in = SpikeGeneratorGroup(n_inputs, indices, inputs[0, 0, :])

# LiF model based on SRM model with alpha function -> a(t) = t/tau * e^(1-t/tau) (see Bohte et al. 2002)
eqs = '''
dv/dt = (x-v)/tau : 1
dx/dt = -x/tau    : 1
'''
on_pre = 'x += w'

# Create rbf neurons
neurons_rbf = NeuronGroup(n_rbf, eqs, threshold='v > Vt', refractory=20 * ms, method='euler')

# Create STDP synapses and connect in/rbf neurons
syn_stdp = Synapses(neurons_in, neurons_rbf,
                    model='''w : 1
                            ti : 1
                            tj : 1
                            delta_t : 1''',
                    on_pre='''x += w
                            ti = t/(1*ms)
                            delta_t = ti - tj - d_back
                            dw = clip(w + lr * ((1-b) * exp(-((delta_t-c)**2)/beta**2) + b),0,w_max) * int(-15 <= delta_t) * int(delta_t <= 15) + w * int(delta_t < -15) + w * int(delta_t > 15)
                            w = int(ti > tj) * int(tj > 0) * dw + int(ti <= tj) * w + int(ti > tj) * int(tj == 0) * w
                          ''',
                    on_post='''
                            tj = t/(1*ms)
                            delta_t = ti - tj - d_back
                            dw = clip(w + lr * ((1-b) * exp(-((delta_t-c)**2)/beta**2) + b),0,w_max) * int(-15 <= delta_t) * int(delta_t <= 15) + w * int(delta_t < -15) + w * int(delta_t > 15)
                            w = int(ti <= tj) * dw + int(ti > tj) * w''',
                    multisynaptic_index='synapse_number')
syn_stdp.connect(n=n_delays)
syn_stdp.delay = '(synapse_number + 1)*ms'
syn_stdp.w[:, :] = 'rand()*w_max*w_factor'

# Create lateral inhibitory connections
syn_inhib = Synapses(neurons_rbf, neurons_rbf, model='w: 1', on_pre=on_pre)
syn_inhib.connect(condition='i != j')
syn_inhib.delay = 0 * ms
syn_inhib.w[:, :] = w_inh

# Create recurrent inhibitory connections
# syn_inhib_rec = Synapses(neurons_rbf, neurons_rbf, model='w: 1', on_pre=on_pre)
# syn_inhib_rec.connect(condition='i == j')
# syn_inhib_rec.delay = 2 * ms
# syn_inhib_rec.w[:, :] = w_inh

# Create Monitors for data aquisition
M_RBF = StateMonitor(neurons_rbf, 'v', record=True)
M_S = StateMonitor(syn_stdp, True, record=True)
spike_mon = SpikeMonitor(neurons_rbf)

# Plot and print initial data
plot_weights(syn_stdp.w)
# print_debugging_data()

spikes = zeros([n_clusters, n_rbf + 1], dtype=int)

for i_cluster in range(n_clusters):
    print(f'\n\n######## Starting simulation for cluster {i_cluster + 1} #########')
    for i_run in range(runs):
        print(f'######## Starting simulation run number {i_run + 1} #########')

        offset = i_run * runtime + i_cluster * runs * runtime
        cluster_id = np.random.randint(n_clusters) if rand_clusters else i_cluster
        neurons_in.set_spikes(indices, inputs[cluster_id, i_run, :] + offset)
        run(runtime)

        if len(spike_mon.t) > 0:
            for i_rbf in range(n_rbf):
                if len(spike_mon.t[spike_mon.i == i_rbf]) > 0:
                    spikes[cluster_id, i_rbf] += (offset <= spike_mon.t[spike_mon.i == i_rbf][-1] <= offset + runtime)
            spikes[cluster_id, n_rbf] += 1

        # print_debugging_data()

    plot_weights(syn_stdp.w)

plot_potential(M_RBF, spike_mon)
# plot_rel_spikes()

# Print spike times for rbf neurons
for i_rbf in range(n_rbf):
    print(f'Spike Times [{i_rbf}] : {(spike_mon.t[spike_mon.i == i_rbf] / ms) % (runtime / ms)}')

# Print table including spike correlations of input and rbf neurons
cols = [''] + [f'neurons_rbf {i}' for i in range(n_rbf)] + ['Total']
t = PrettyTable(cols)
for i_c in range(n_clusters):
    row = [f'Cluster {centers[i_c]}']
    for i_rbf in range(n_rbf + 1):
        row.append(spikes[i_c, i_rbf])
    t.add_row(row)
print(t)
