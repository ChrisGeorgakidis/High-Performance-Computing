import numpy

fd = open("Tiled_with_padding/doubles/tiled_with_padding_doubles_64_8192x8192.txt", "r+")

lines = [x.strip('\n') for x in fd.readlines()]

cpu_list = [float(lines[x].split()[3]) for x in range(8,200,17)]
gpu_list = [float(lines[x].split()[3])*(10**(-3)) for x in range(12,200,17)]

print (min(cpu_list))
print (max(cpu_list))
print ('----->  <-------')

del cpu_list[cpu_list.index(max(cpu_list))]
del cpu_list[cpu_list.index(min(cpu_list))]

for number in cpu_list:
    print(number)

print('------------')
print (numpy.mean(cpu_list, axis=0))
print (numpy.std(cpu_list, axis=0))

print('----------')
print (min(gpu_list))
print (max(gpu_list))
#print ('----->  <-------')

del gpu_list[gpu_list.index(max(gpu_list))]
del gpu_list[gpu_list.index(min(gpu_list))]
for number in gpu_list:
    print(number)
#print('----------')
print (numpy.mean(gpu_list, axis=0))
print (numpy.std(gpu_list, axis=0))
