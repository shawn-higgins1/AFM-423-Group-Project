�	ؼ��Z\7@ؼ��Z\7@!ؼ��Z\7@	��W?uh�?��W?uh�?!��W?uh�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6ؼ��Z\7@�q���4@1cE�a��?A�C���?I��Ye�T@Y�R�t?*	     �K@2U
Iterator::Model::ParallelMapV2��ׁsF�?!h�`�|�A@)��ׁsF�?1h�`�|�A@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat �o_Ή?!?�S�6@)/�$��?1�n0E>�2@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateǺ����?!�B�I .4@)ŏ1w-!?12���+c+@:Preprocessing2F
Iterator::ModelS�!�uq�?!Oq��$H@)y�&1�|?1��@\�9)@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!#�u�)�@)��H�}m?1#�u�)�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipO��e�c�?!���^�I@)a��+ei?1zel��W@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!T�n0E@)�J�4a?1T�n0E@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap9��v���?!��;zel7@)��H�}]?1#�u�)�	@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��W?uh�?Id-9P�X@Q�G�\\��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�q���4@�q���4@!�q���4@      ��!       "	cE�a��?cE�a��?!cE�a��?*      ��!       2	�C���?�C���?!�C���?:	��Ye�T@��Ye�T@!��Ye�T@B      ��!       J	�R�t?�R�t?!�R�t?R      ��!       Z	�R�t?�R�t?!�R�t?b      ��!       JGPUY��W?uh�?b qd-9P�X@y�G�\\��?�"5
sequential/dense/MatMulMatMulJ?=+��?!J?=+��?0"C
%gradient_tape/sequential/dense/MatMulMatMul��R_�U�?!H�"k�?0"7
sequential/dense_1/MatMulMatMul��R_�U�?!�����?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�&V�?!��R_�U�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�&V�?!�-ĉ �?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMulcT��� �?!���`�?"7
sequential/dense_2/MatMulMatMul��'�7��?!�����?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�-ĉ �?!##w-��?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGradظ��V�?!&>����?"]
?gradient_tape/mean_squared_error/weighted_loss/value/div_no_nanDivNoNan�>�G��?!	a����?Q      Y@Y>����/@aX�i��U@q���[5W@y�����?"�
both�Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�92.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 