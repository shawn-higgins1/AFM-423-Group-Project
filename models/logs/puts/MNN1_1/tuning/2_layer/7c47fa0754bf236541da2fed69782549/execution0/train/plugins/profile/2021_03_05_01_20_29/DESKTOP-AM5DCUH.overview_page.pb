�	5@i�QX7@5@i�QX7@!5@i�QX7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-5@i�QX7@��cw�4@1Y�E����?A�,C��?IY��9� @*	     �J@2U
Iterator::Model::ParallelMapV2�q����?!���B�(=@)�q����?1���B�(=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�]K�=�?!A�Ե�8@)A��ǘ��?1����f�4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate-C��6�?!�Ե��7@)/n���?1e�S�r0@:Preprocessing2F
Iterator::Model��0�*�?!M�w�ZF@)����Mb�?1*J�#�-@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!*J�#�@)����Mbp?1*J�#�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipu���?!�_���K@)�����g?1��z��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!e�S�r@)/n��b?1e�S�r@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�?�߾�?!靺���9@)��H�}M?1��L�w��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��+��X@Q�U u>B�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��cw�4@��cw�4@!��cw�4@      ��!       "	Y�E����?Y�E����?!Y�E����?*      ��!       2	�,C��?�,C��?!�,C��?:	Y��9� @Y��9� @!Y��9� @B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��+��X@y�U u>B�?�"C
%gradient_tape/sequential/dense/MatMulMatMul�9rb�a�?!�9rb�a�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�9rb�a�?!�9rb�a�?0"5
sequential/dense/MatMulMatMul�9rb�a�?!�V�Q�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulH�J�f-�?!Jd�9�T�?"7
sequential/dense_1/MatMulMatMulm�G�j�?!�`q��!�?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrads�;?!:�~��?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMuls�;?!5FT��?"!
Adam/PowPow�9rb�a�?!SZ�����?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�9rb�a�?!������?"7
sequential/dense_2/MatMulMatMul�9rb�a�?!h�Et��?0Q      Y@Y>����/@aX�i��U@q��}�+rX@y�z��i(�?"�
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
Refer to the TF2 Profiler FAQb�97.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 