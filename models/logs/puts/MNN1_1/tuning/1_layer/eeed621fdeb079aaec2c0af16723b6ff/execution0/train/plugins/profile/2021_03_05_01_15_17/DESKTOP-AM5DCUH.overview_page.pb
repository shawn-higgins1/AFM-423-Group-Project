�	E-ͭ�5@E-ͭ�5@!E-ͭ�5@	�:����?�:����?!�:����?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6E-ͭ�5@/��$3@1��z�2Q�?A+��Χ?IJ}Yک� @Y{k`���?*	33333�J@2U
Iterator::Model::ParallelMapV2	�^)ː?!>G�D=m>@)	�^)ː?1>G�D=m>@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�ZӼ��?!97dWX:@)��ZӼ�?1���2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeata��+e�?!�N��l7@)��ׁsF�?1�S�r
^2@:Preprocessing2F
Iterator::Model�+e�X�?!�Pb�x&E@)-C��6z?1|��h�'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!��B�@)����Mbp?1��B�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�U���؟?!A��\��L@)�~j�t�h?14鏃qC@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor{�G�zd?! ��B��@){�G�zd?1 ��B��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap2U0*��?!`��� =@)�~j�t�X?14鏃qC@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�:����?I�n)��X@Q�+�����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	/��$3@/��$3@!/��$3@      ��!       "	��z�2Q�?��z�2Q�?!��z�2Q�?*      ��!       2	+��Χ?+��Χ?!+��Χ?:	J}Yک� @J}Yک� @!J}Yک� @B      ��!       J	{k`���?{k`���?!{k`���?R      ��!       Z	{k`���?{k`���?!{k`���?b      ��!       JGPUY�:����?b q�n)��X@y�+�����?�"5
sequential/dense/MatMulMatMul��wg��?!��wg��?0"C
%gradient_tape/sequential/dense/MatMulMatMul�Q�Jgf�?!M�,Y��?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch����?!�����?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul����?!����6�?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�Q�Jgf�?!��X�\��?"!
Adam/PowPow�Q�Jgf�?!��)P�?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�Q�Jgf�?!N1����?"7
sequential/dense_1/MatMulMatMul�Q�Jgf�?!�[m��i�?0"O
5gradient_tape/mean_squared_error/weighted_loss/Tile_1Tile�V;�'��?!�A�S�?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam��Q𚙎?!�0F�=�?Q      Y@Y��/Ċ�0@a�	�N]�T@q�<�E�lW@y� e��?"�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.8 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�93.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 