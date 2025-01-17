�	[��YeR7@[��YeR7@![��YeR7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-[��YeR7@��Q��4@19�)9'��?A�Z}uU��?I��,��@*	����̌K@2U
Iterator::Model::ParallelMapV2��ǘ���?!,əíf=@)��ǘ���?1,əíf=@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenateŏ1w-!�?!a���;@)Zd;�O��?1֬�@�4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat �o_Ή?!�>_�6@)/�$��?1�D�$g3@:Preprocessing2F
Iterator::Model�+e�X�?!��=��D@)F%u�{?1���o��'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vOn?!ђ�9�@)���_vOn?1ђ�9�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipaTR'���?!+�4�rOM@)y�&1�l?1�E��h@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!8�S�q}@)�J�4a?18�S�q}@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap��ǘ���?!,əíf=@)����MbP?1*���	�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��g�X@Q�	&9��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��Q��4@��Q��4@!��Q��4@      ��!       "	9�)9'��?9�)9'��?!9�)9'��?*      ��!       2	�Z}uU��?�Z}uU��?!�Z}uU��?:	��,��@��,��@!��,��@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��g�X@y�	&9��?�"5
sequential/dense/MatMulMatMul���X��?!���X��?0"C
%gradient_tape/sequential/dense/MatMulMatMul��f�?!^ZZ	�|�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�ݥ�Nf�?!RI-M 0�?0"7
sequential/dense_1/MatMulMatMul�ݥ�Nf�?!#�ȣ��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulWDbm	s�?!�d,���?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradcU�)⿕?!ZO^;x�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMulcU�)⿕?!:���/�?"7
sequential/dense_2/MatMulMatMulcU�)⿕?!�$�����?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�ݥ�Nf�?!8p[�a
�?"E
'gradient_tape/sequential/dense_2/MatMulMatMul�ݥ�Nf�?!�ճ� �?0Q      Y@Y>����/@aX�i��U@q��Ŋ^X@y�y�Q�-�?"�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 