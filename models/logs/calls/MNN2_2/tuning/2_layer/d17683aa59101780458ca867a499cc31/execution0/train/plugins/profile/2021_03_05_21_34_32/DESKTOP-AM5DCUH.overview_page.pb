�	Ym�_u�D@Ym�_u�D@!Ym�_u�D@	�֘^\`�?�֘^\`�?!�֘^\`�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6Ym�_u�D@�lscz�B@1�"�k$�?Aq���h�?I����W
@Y�����?*	effff�K@2U
Iterator::Model::ParallelMapV2����Mb�?!�9�!��<@)����Mb�?1�9�!��<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��ǘ���?!�*uK=@)�Pk�w�?1EZ�>�"9@:Preprocessing2F
Iterator::ModeltF��_�?!
�Z܄E@)�q����?1lXY'�5,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate��_�L�?!��ow�2@)F%u�{?1��H��'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceŏ1w-!o?!�v�,�|@)ŏ1w-!o?1�v�,�|@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipT㥛� �?!�D�#{L@)��H�}m?1�8/
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!FA;��@)HP�s�b?1FA;��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapg��j+��?!Q�]F(5@)��_�LU?1��ow�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 91.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�֘^\`�?I�L��~�X@Qr�����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�lscz�B@�lscz�B@!�lscz�B@      ��!       "	�"�k$�?�"�k$�?!�"�k$�?*      ��!       2	q���h�?q���h�?!q���h�?:	����W
@����W
@!����W
@B      ��!       J	�����?�����?!�����?R      ��!       Z	�����?�����?!�����?b      ��!       JGPUY�֘^\`�?b q�L��~�X@yr�����?�"5
sequential/dense/MatMulMatMul�kL��d�?!�kL��d�?0"C
%gradient_tape/sequential/dense/MatMulMatMul��ʎ�=�?!*��[Q�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul��ʎ�=�?!�qB�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul���B=�?!���q��?"7
sequential/dense_1/MatMulMatMul���B=�?!�bX��?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMulGM4gtt�?!l���PE�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchD"�M���?!�P�N���?"7
sequential/dense_2/MatMulMatMulD"�M���?!��r�u��?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad���B=�?!ε�h�?")
sequential/CastCast���B=�?!�0�{�?Q      Y@Y>����/@aX�i��U@q�
6Zp�U@y ��3�?"�
both�Your program is POTENTIALLY input-bound because 91.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�8.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�87.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 