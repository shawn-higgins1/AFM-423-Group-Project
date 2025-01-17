�	���7@���7@!���7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-���7@+�� .4@1X)�k{�?AǺ���?IeT��`@*	effff&M@2U
Iterator::Model::ParallelMapV2�Q���?!?��d>@)�Q���?1?��d>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat��H�}�?!�v�'�8@)��@��ǈ?1���!5�4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate2U0*��?!f:-��:@)g��j+��?1��ow�4@:Preprocessing2F
Iterator::Model��+e�?!��)�D@)_�Q�{?1A����S'@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!����q@)����Mbp?1����q@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�� �rh�?!V�H�(M@)�����g?1�
�L��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHP�s�b?!*`3���@)HP�s�b?1*`3���@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�o_��?!��<@)����MbP?1����q�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�\e��X@Q�(���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	+�� .4@+�� .4@!+�� .4@      ��!       "	X)�k{�?X)�k{�?!X)�k{�?*      ��!       2	Ǻ���?Ǻ���?!Ǻ���?:	eT��`@eT��`@!eT��`@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�\e��X@y�(���?�"5
sequential/dense/MatMulMatMuluWdv7�?!uWdv7�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�Z�؞�?!"٭'k�?0"7
sequential/dense_1/MatMulMatMul�Z�؞�?!��)�:��?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul0+���?!ڳc��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�ް��ם?!�)Ps^D�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�15�Q�?! �6E��?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�15�Q�?!Pv���?"7
sequential/dense_2/MatMulMatMul�15�Q�?!���B�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�Z�؞�?!�#�td�?"E
'gradient_tape/sequential/dense_2/MatMulMatMul�Z�؞�?!���O��?0Q      Y@Y>����/@aX�i��U@q;P�N�4W@y�}3H��?"�
both�Your program is POTENTIALLY input-bound because 87.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�92.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 