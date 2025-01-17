�	��G��8@��G��8@!��G��8@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��G��8@���W�5@1'ݖ�g�?A9��m4��?I��
؎@*	�����YJ@2U
Iterator::Model::ParallelMapV29��v���?!�`7���8@)9��v���?1�`7���8@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat%u��?!�G���;@)�(��0�?1W���V7@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate��<,Ԋ?!��>��8@)a2U0*��?17���`72@:Preprocessing2F
Iterator::Model��~j�t�?!���B@)�~j�t�x?1����8�&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip7�[ A�?!�u3�O@)/n��r?1�phGò @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlicey�&1�l?!�ʱN@)y�&1�l?1�ʱN@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensora2U0*�c?!7���`7@)a2U0*�c?17���`7@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�<,Ԛ�?!��aY8�;@)�~j�t�X?1����8�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIVv�*�X@Q�|j"Z5�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���W�5@���W�5@!���W�5@      ��!       "	'ݖ�g�?'ݖ�g�?!'ݖ�g�?*      ��!       2	9��m4��?9��m4��?!9��m4��?:	��
؎@��
؎@!��
؎@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qVv�*�X@y�|j"Z5�?�"5
sequential/dense/MatMulMatMul:ou'h�?!:ou'h�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul!&����?!��D
	�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�#@��?!rQ����?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�#@��?!�,��?"7
sequential/dense_1/MatMulMatMul�#@��?!}�p|��?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMulڿ�8�?!��k�f�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch���+�7�?!�����?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad���+�7�?!��a"t��?"7
sequential/dense_2/MatMulMatMul���+�7�?!o��-�?0"!
Adam/PowPow�#@��?!�G=�?Q      Y@Y>����/@aX�i��U@q�ypX@y$��6t��?"�
both�Your program is POTENTIALLY input-bound because 87.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 