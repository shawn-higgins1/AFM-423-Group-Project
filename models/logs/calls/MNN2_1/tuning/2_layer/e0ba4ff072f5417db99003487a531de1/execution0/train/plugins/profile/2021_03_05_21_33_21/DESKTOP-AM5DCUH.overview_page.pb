�	6 B\9�7@6 B\9�7@!6 B\9�7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-6 B\9�7@����y5@1�2����?A�+e�X�?I=((E+@*	������K@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate���Mb�?!�m۶m�D@)w-!�l�?1۶m۶�C@:Preprocessing2U
Iterator::Model::ParallelMapV2�������?!%I�$IR6@)�������?1%I�$IR6@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�� �rh�?!�m۶m[.@)F%u�{?1$I�$I�'@:Preprocessing2F
Iterator::Model�ݓ��Z�?!     �@@)-C��6z?1�m۶m�&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��\m���?!     �P@)ŏ1w-!o?1I�$I�$@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!I�$I�$@)ŏ1w-!_?1I�$I�$@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapa��+e�?!I�$I�$F@)��_�LU?1%I�$I�@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor/n��R?!۶m۶m�?)/n��R?1۶m۶m�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice����Mb@?!$I�$I��?)����Mb@?1$I�$I��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI
���X@Q�=O���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����y5@����y5@!����y5@      ��!       "	�2����?�2����?!�2����?*      ��!       2	�+e�X�?�+e�X�?!�+e�X�?:	=((E+@=((E+@!=((E+@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q
���X@y�=O���?�"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��zy��?!��zy��?"5
sequential/dense/MatMulMatMul��|⛚�?!̤�-%θ?0"C
%gradient_tape/sequential/dense/MatMulMatMul3çn��?!3çn��?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul3çn��?! �Q�ɫ�?0"7
sequential/dense_1/MatMulMatMul������?!�f��I�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch��Q�ɫ�?!3���R��?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad��Q�ɫ�?!�Ӛ&�t�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul��Q�ɫ�?!���"�?"7
sequential/dense_2/MatMulMatMul����v��?!���?0"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam���"�?!FH�C�?Q      Y@Y�C=�C=0@a��
��T@q��_phyX@y���a�?"�
both�Your program is POTENTIALLY input-bound because 89.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�8.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 