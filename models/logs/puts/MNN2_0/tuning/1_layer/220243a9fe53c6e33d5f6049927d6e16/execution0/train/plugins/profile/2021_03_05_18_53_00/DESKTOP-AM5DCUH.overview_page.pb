�	g�\S W6@g�\S W6@!g�\S W6@	��� ��?��� ��?!��� ��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6g�\S W6@��Ĭ4@1�*��	�?Ag��j+��?I �.�� @Y�sa���?*	hfffffK@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�z6�>�?!t2}�0�D@)/�$��?1�uy)C@:Preprocessing2U
Iterator::Model::ParallelMapV2?W[���?!�ܺ��;@)?W[���?1�ܺ��;@:Preprocessing2F
Iterator::Model46<�R�?!G����C@)S�!�uq{?1
c���s(@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�~j�t�x?!������%@)�q����o?1�[�w@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipL7�A`�?!��uyN@)-C��6j?1)�[@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!�U��,�@)�J�4a?1�U��,�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMaptF��_�?!8R4��E@)/n��R?1C�q�� @:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor/n��R?!C�q�� @)/n��R?1C�q�� @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!՞�髄�?)a2U0*�C?1՞�髄�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��� ��?I�fh8�X@QcQDi#�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��Ĭ4@��Ĭ4@!��Ĭ4@      ��!       "	�*��	�?�*��	�?!�*��	�?*      ��!       2	g��j+��?g��j+��?!g��j+��?:	 �.�� @ �.�� @! �.�� @B      ��!       J	�sa���?�sa���?!�sa���?R      ��!       Z	�sa���?�sa���?!�sa���?b      ��!       JGPUY��� ��?b q�fh8�X@ycQDi#�?�"5
sequential/dense/MatMulMatMul�\�2�?!�\�2�?0"C
%gradient_tape/sequential/dense/MatMulMatMulڤ-���?!րQ�ݵ?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��u��ř?!��5�`O�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch(a����?!�P�E[�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad6�cQ��?!��¯��?"7
sequential/dense_1/MatMulMatMul6�cQ��?!�
5�ق�?0"!
Adam/PowPow_�t�^��?!�C��?"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam�g�妓?!m�i�v�?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam�Yz�?!
�
�e�?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam�Yz�?!&�K��T�?Q      Y@Y{	�%��1@a�����T@qsp�61W@y<rɕ��?"�
both�Your program is POTENTIALLY input-bound because 89.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�92.3% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 