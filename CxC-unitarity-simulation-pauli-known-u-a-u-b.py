import qutip as qt
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# UNITARITIES and SET-UP
g_1Q=[qt.identity(2), qt.sigmax(), qt.sigmay(), qt.sigmaz()]
g_2Q_A=[qt.tensor(g_1Q[1],g_1Q[0])/2,qt.tensor(g_1Q[2],g_1Q[0])/2,qt.tensor(g_1Q[3],g_1Q[0])/2]
g_2Q_B=[qt.tensor(g_1Q[0],g_1Q[1])/2,qt.tensor(g_1Q[0],g_1Q[2])/2,qt.tensor(g_1Q[0],g_1Q[3])/2]
g_2Q_all=[qt.tensor(g_1Q[1],g_1Q[1])/2, qt.tensor(g_1Q[1],g_1Q[2])/2, qt.tensor(g_1Q[1],g_1Q[3])/2, qt.tensor(g_1Q[2],g_1Q[1])/2, qt.tensor(g_1Q[2],g_1Q[2])/2, qt.tensor(g_1Q[2],g_1Q[3])/2, qt.tensor(g_1Q[3],g_1Q[1])/2, qt.tensor(g_1Q[3],g_1Q[2])/2, qt.tensor(g_1Q[3],g_1Q[3])/2,]
g_2Q_all_all = [qt.tensor(g_1Q[1],g_1Q[0])/2,qt.tensor(g_1Q[2],g_1Q[0])/2,qt.tensor(g_1Q[3],g_1Q[0])/2,qt.tensor(g_1Q[1],g_1Q[1])/2, qt.tensor(g_1Q[1],g_1Q[2])/2, qt.tensor(g_1Q[1],g_1Q[3])/2, qt.tensor(g_1Q[2],g_1Q[1])/2, qt.tensor(g_1Q[2],g_1Q[2])/2, qt.tensor(g_1Q[2],g_1Q[3])/2, qt.tensor(g_1Q[3],g_1Q[1])/2, qt.tensor(g_1Q[3],g_1Q[2])/2, qt.tensor(g_1Q[3],g_1Q[3])/2,qt.tensor(g_1Q[0],g_1Q[1])/2,qt.tensor(g_1Q[0],g_1Q[2])/2,qt.tensor(g_1Q[0],g_1Q[3])/2]

def hilbert_shm(the_bra,the_ket):
    hilb_sm=the_bra.dag() * the_ket
    return hilb_sm.tr()

def liouville_element(channel,the_bra,the_ket):
    return hilbert_shm(the_bra, channel(the_ket))

def pauli_channel_2_qubit(num_of_paulis):
    paulis_probs=np.random.random_sample(num_of_paulis)
    paulis_probs = paulis_probs / sum(paulis_probs)
    channel=qt.Qobj()
    for i in range(num_of_paulis):
        pauli_rand = qt.tensor(g_1Q[rd.randint(0,3)],g_1Q[rd.randint(0,3)])
        channel += paulis_probs[i] * qt.sprepost(pauli_rand,pauli_rand)
    return channel

# -------- All the unitarities ------------

def sub_unitarity_A_to_A(channel):
    T_a=np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            T_a[i,j]=liouville_element(channel,g_2Q_A[i],g_2Q_A[j])
    T_a=qt.Qobj(T_a)
    T_a_sqr=T_a.dag() * T_a
    return (1./(4. - 1.)) * T_a_sqr.tr()

def sub_unitarity_B_to_B(channel):
    T_a=np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            T_a[i,j]=liouville_element(channel,g_2Q_B[i],g_2Q_B[j])
    T_a=qt.Qobj(T_a)
    T_a_sqr=T_a.dag() * T_a
    return (1./(4. - 1.)) * T_a_sqr.tr()


def sub_unitarity_A_to_B(channel):
    T_a=np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            T_a[i,j]=liouville_element(channel,g_2Q_B[i],g_2Q_A[j])
    T_a=qt.Qobj(T_a)
    T_a_sqr=T_a.dag() * T_a
    return (1./(4. - 1.)) * T_a_sqr.tr()

def sub_unitarity_B_to_A(channel):
    T_a=np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            T_a[i,j]=liouville_element(channel,g_2Q_A[i],g_2Q_B[j])
    T_a=qt.Qobj(T_a)
    T_a_sqr=T_a.dag() * T_a
    return (1./(4. - 1.)) * T_a_sqr.tr()


def sub_unitarity_AB_to_AB(channel):
    T_a=np.zeros((9, 9), dtype=complex)
    for i in range(9):
        for j in range(9):
            T_a[i,j]=liouville_element(channel, g_2Q_all[i], g_2Q_all[j])
    T_a=qt.Qobj(T_a)
    T_a_sqr=T_a.dag() * T_a
    return (1./(10. - 1.)) * T_a_sqr.tr()
    

def sub_unitarity_A_to_AB(channel):
    T_a=np.zeros((9, 3), dtype=complex)
    for i in range(9):
        for j in range(3):
            T_a[i,j]=liouville_element(channel, g_2Q_all[i], g_2Q_A[j])
    T_a=qt.Qobj(T_a)
    T_a_sqr=T_a.dag() * T_a
    return (1./(np.sqrt(27.))) * T_a_sqr.tr()
    

def sub_unitarity_B_to_AB(channel):
    T_a=np.zeros((9, 3), dtype=complex)
    for i in range(9):
        for j in range(3):
            T_a[i,j]=liouville_element(channel, g_2Q_all[i], g_2Q_B[j])
    T_a=qt.Qobj(T_a)
    T_a_sqr=T_a.dag() * T_a
    return (1./(np.sqrt(27.))) * T_a_sqr.tr()
    

def sub_unitarity_AB_to_A(channel):
    T_a=np.zeros((3, 9), dtype=complex)
    for i in range(3):
        for j in range(9):
            T_a[i,j]=liouville_element(channel, g_2Q_A[i], g_2Q_all[j])
    T_a=qt.Qobj(T_a)
    T_a_sqr=T_a.dag() * T_a
    return (1./(np.sqrt(27.))) * T_a_sqr.tr()
    

def sub_unitarity_AB_to_B(channel):
    T_a=np.zeros((3, 9), dtype=complex)
    for i in range(3):
        for j in range(9):
            T_a[i,j]=liouville_element(channel, g_2Q_B[i], g_2Q_all[j])
    T_a=qt.Qobj(T_a)
    T_a_sqr=T_a.dag() * T_a
    return (1./(np.sqrt(27.))) * T_a_sqr.tr()

# ~~~~~ Correlation measures ~~~~~~

def matrixA(channel):
    A=np.zeros((3, 3), dtype=complex)
    A[0,0]=sub_unitarity_A_to_A(channel)
    A[0,1]=sub_unitarity_AB_to_A(channel)
    A[0,2]=sub_unitarity_B_to_A(channel)
    A[1,0]=sub_unitarity_A_to_AB(channel)
    A[1,1]=sub_unitarity_AB_to_AB(channel)
    A[1,2]=sub_unitarity_B_to_AB(channel)
    A[2,0]=sub_unitarity_A_to_B(channel)
    A[2,1]=sub_unitarity_AB_to_B(channel)
    A[2,2]=sub_unitarity_B_to_B(channel)
    return A

def eigens_of_A(channel): 
    matA = matrixA(channel)
    eigens=np.linalg.eigvals(matA)
    eigens=np.real(eigens)
    eigens=np.sort(eigens)
    return eigens

def eigen_measure(channel): #largest first
    sorted_eigens = eigens_of_A(channel)
    return (sorted_eigens[2] - (sorted_eigens[0])*(sorted_eigens[1]))

def correlated_unitarity(channel):
 return (sub_unitarity_AB_to_AB(channel) - (sub_unitarity_A_to_A(channel))*(sub_unitarity_B_to_B(channel)))

def eigen_measure_fix(channel):  #smallest first
    sorted_eigens = eigens_of_A(channel)
    measure = (sorted_eigens[0] - (sorted_eigens[2])*(sorted_eigens[1]))
    return measure

# SIMULATIONS

def unitary_channel(dimension,desired_fidelity,error):
    noise_operator = qt.to_super(qt.rand_unitary(dimension))
    count=0
    while np.abs(qt.average_gate_fidelity(noise_operator)-desired_fidelity)>error and count<400:
        noise_operator = qt.to_super(qt.rand_unitary(dimension))
        count=count+1
    if count>=400: 
        print("Could not achieve desired fidelity within margin of error")
    return noise_operator 

# def channel(dimension,desired_fidelity,error):
#     noise_operator = qt.rand_super(dimension)
#     count=0
#     while np.abs(qt.average_gate_fidelity(noise_operator)-desired_fidelity)>error and count<400:
#         noise_operator = qt.rand_super(dimension)
#         count=count+1
#     if count>=400: 
#         print("Could not achieve desired fidelity within margin of error")
#     return noise_operator

def func(m,c0,c1,c2,c3,e1,e2,e3):
    return (c0 + c1* e1**(m-1.) + c2* e2**(m-1.) + c3* e3**(m-1.) )

#This function performs the unitarity CxC randomized benchmarking
def protocol(cliffords,rho, Q, max_sequence_length,number_of_sequences):
    outcome_array = np.empty([max_sequence_length-1])

    for ii in range(0, max_sequence_length-1):
        kk=0
        a = np.empty([number_of_sequences])

        while kk < number_of_sequences:
            density = rho

            for jj in range(0, ii+1):
                noisy_super_clifford = random_noisy_clifford()
                density = noisy_super_clifford*density
                # if ii ==0:
                #     print 'yes'

            A = Q.dag()*density
            a[kk] = (A.tr())**2
            kk = kk+1  

        outcome_array[ii]=np.mean(a)
    return outcome_array


f = plt.figure(figsize=(15,10))
ax = f.add_subplot(111)

num_of_data_points=20
for i in range(num_of_data_points):
    print "Run: ", i
    # ~~~~~~~~~~ Set the noise channel here ~~~~~~~~~~~
    found_a_good_channel = False
    limit = 0.3
    while found_a_good_channel == False:
        # two_qubit_noise = unitary_channel(4,0.9,0.01) # A unitary
        # two_qubit_noise = qt.to_super(qt.cnot()) # CNOT
        # two_qubit_noise = qt.rand_super_bcsz(4, dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]], rank=2)
        # two_qubit_noise = qt.rand_super_bcsz(4, dims=[[[2, 2], [2, 2]], [[2, 2], [2, 2]]])
        two_qubit_noise = pauli_channel_2_qubit(4)
        # two_qubit_noise = qt.super_tensor(qt.rand_super(2),qt.rand_super(2))
        # same=qt.rand_super(2)
        # two_qubit_noise = qt.super_tensor(same,same)
        if qt.unitarity(two_qubit_noise) > limit:
            found_a_good_channel = True

    two_qubit_noise.dims = [[[2, 2], [2, 2]], [[2, 2], [2, 2]]]


    # Prints information about the noise (this is helpful to see if the channel is suitable)
    print "Unitarity: ", qt.unitarity(two_qubit_noise)
    print "Correlated Unitarity: ", correlated_unitarity(two_qubit_noise)
    print "True sub-unitarities:"
    print " AB->AB: ", sub_unitarity_AB_to_AB(two_qubit_noise), "A->A: ", sub_unitarity_A_to_A(two_qubit_noise), " B->B: ", sub_unitarity_B_to_B(two_qubit_noise)

    # Makes the gates noisy
    noisy_cliffords=[two_qubit_noise*qt.to_super(qt.tensor(x,y)) for x in qt.qubit_clifford_group() for y in qt.qubit_clifford_group()]
    def random_noisy_clifford():
            rand_int = rd.randint(0,len(noisy_cliffords)-1)
            return noisy_cliffords[rand_int]

    # Some SPAM
    noise_operator_state_prep =unitary_channel(4,0.8,0.01)
    noise_operator_meas_op = unitary_channel(4,0.8,0.01)
    noise_operator_state_prep.dims = [[[2, 2], [2, 2]], [[2, 2], [2, 2]]]
    noise_operator_meas_op.dims = [[[2, 2], [2, 2]], [[2, 2], [2, 2]]]

    # Set the number of runs
    number_of_seq = 5000
    seq_length = 25

    # Set observable Q
    Q=qt.tensor(qt.sigmay()-qt.sigmaz(),qt.sigmay()-qt.sigmaz())
    Q.dims = [[2,2],[2,2]]
    Q =qt.operator_to_vector(Q)
    Q = noise_operator_meas_op*Q

    # Set initial state
    rho = qt.basis(4,0)*qt.basis(4,0).dag()
    rho.dims = [[2,2],[2,2]]
    rho = qt.operator_to_vector(rho)
    rho = noise_operator_state_prep*rho

    def fixed_func(m,c0,c1,c2,c3,e1):
        return (c0 + c1*e1**(m-1.) + c2* (sub_unitarity_A_to_A(two_qubit_noise))**(m-1.) + c3* (sub_unitarity_B_to_B(two_qubit_noise))**(m-1.) )

    # Run!
    reference_standard_data = protocol(noisy_cliffords,rho,Q,seq_length,number_of_seq)

    reference_optimal_standard = opt.curve_fit(fixed_func,np.arange(1,seq_length),reference_standard_data,p0=[0.,0.2,0.2,0.2,0.5], bounds=([0.,0.,0.,0.,0.], 1.) )

    print "Simulated information: "
    print "c0: ", reference_optimal_standard[0][0], " c1: ", reference_optimal_standard[0][1]," c2: ", reference_optimal_standard[0][2] , " c3: ", reference_optimal_standard[0][3]
    print "AB->AB (free parameter): ", reference_optimal_standard[0][4], " A->A (known): ", sub_unitarity_A_to_A(two_qubit_noise), " B->B (known): ", sub_unitarity_B_to_B(two_qubit_noise)
    print "Correlated unitarity: ", (reference_optimal_standard[0][4] -  sub_unitarity_A_to_A(two_qubit_noise)*sub_unitarity_B_to_B(two_qubit_noise))

    with open("RB simulations/pauli-channel-uab.txt", "a") as file_object:
        file_object.write("\n%f %f" % (sub_unitarity_AB_to_AB(two_qubit_noise), reference_optimal_standard[0][4]))

# Plotting.
ax.scatter(range(1,seq_length),reference_standard_data,marker = 'o', label='Data')
ax.plot(range(1,seq_length),fixed_func(np.arange(1,seq_length),reference_optimal_standard[0][0],reference_optimal_standard[0][1],reference_optimal_standard[0][2],reference_optimal_standard[0][3],reference_optimal_standard[0][4]),label='Fit', color='orange')
ax.set_xlabel('Sequence length m',fontsize=18)
ax.set_ylabel('Average value of $(Q)^2$',fontsize=18)
ax.set_title('CxC two-qubit Unitarity RB',fontsize=25)
ax.legend(fontsize = 15)
# plt.show()

