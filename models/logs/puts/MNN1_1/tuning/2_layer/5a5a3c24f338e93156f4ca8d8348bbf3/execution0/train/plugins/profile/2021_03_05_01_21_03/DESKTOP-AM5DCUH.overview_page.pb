�	��A�p�6@��A�p�6@!��A�p�6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��A�p�6@��E��L4@1m7�7M��?AZd;�O��?IU3k) -@*	     �J@2U
Iterator::Model::ParallelMapV2�� �rh�?!�~!V��?@)�� �rh�?1�~!V��?@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�{�Pk�?!�@�Ե8@)�I+��?1�B�(��4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate�
F%u�?!Ei|d�7@)�5�;Nс?1�����B0@:Preprocessing2F
Iterator::ModeltF��_�?!���4>F@)_�Q�{?1>2�ީk)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!*J�#�@)����Mbp?1*J�#�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���B�i�?!
�N]��K@)�����g?1��z��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!��bEi@)ŏ1w-!_?1��bEi@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�Pk�w�?!<u-7�9@)a2U0*�S?1�_���@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI1c���X@Q�3�X��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��E��L4@��E��L4@!��E��L4@      ��!       "	m7�7M��?m7�7M��?!m7�7M��?*      ��!       2	Zd;�O��?Zd;�O��?!Zd;�O��?:	U3k) -@U3k) -@!U3k) -@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q1c���X@y�3�X��?�"5
sequential/dense/MatMulMatMul��]﨣?!��]﨣?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��p��y�?!e���[��?"C
%gradient_tape/sequential/dense/MatMulMatMul0+�<y�?!��Q��M�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul0+�<y�?!���K�?0"7
sequential/dense_1/MatMulMatMul0+�<y�?!��pۚc�?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�N���ؕ?!v��0��?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�N���ؕ?!VQޅ���?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�5��ו?!؜贔�?"7
sequential/dense_2/MatMulMatMul��(��L�?!��� (��?0"O
5gradient_tape/mean_squared_error/weighted_loss/Tile_1Tile0+�<y�?!�������?Q      Y@Y>����/@aX�i��U@q�,��mX@y{Cq�mM�?"�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 