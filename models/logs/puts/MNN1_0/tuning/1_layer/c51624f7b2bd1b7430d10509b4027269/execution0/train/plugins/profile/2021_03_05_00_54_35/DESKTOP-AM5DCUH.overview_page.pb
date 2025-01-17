�	��S9�M7@��S9�M7@!��S9�M7@	y'@xq�?y'@xq�?!y'@xq�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6��S9�M7@���
Ԃ4@12Ƈ���?A9��m4��?I�$��G@Y���GS=i?*	      M@2U
Iterator::Model::ParallelMapV2a2U0*��?!���=�@@)a2U0*��?1���=�@@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateV-��?!      9@)�I+��?1�rO#,�2@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat �o_Ή?!�{a�5@)/�$��?1���{2@:Preprocessing2F
Iterator::Model9��v���?!�FX�iF@)_�Q�{?1#,�4�r'@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip����Mb�?!a���K@)y�&1�l?1�4�rO#@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicey�&1�l?!�4�rO#@)y�&1�l?1�4�rO#@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!�rO#,�@)�J�4a?1�rO#,�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap���H�?!�FX�i;@)Ǻ���V?1,�4�rO@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9y'@xq�?I���v��X@Q]84����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���
Ԃ4@���
Ԃ4@!���
Ԃ4@      ��!       "	2Ƈ���?2Ƈ���?!2Ƈ���?*      ��!       2	9��m4��?9��m4��?!9��m4��?:	�$��G@�$��G@!�$��G@B      ��!       J	���GS=i?���GS=i?!���GS=i?R      ��!       Z	���GS=i?���GS=i?!���GS=i?b      ��!       JGPUYy'@xq�?b q���v��X@y]84����?�"5
sequential/dense/MatMulMatMul�~�(��?!�~�(��?0"C
%gradient_tape/sequential/dense/MatMulMatMulTɑ	�?!^��
�I�?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad#�;vI�?!��thj��?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul#�;vI�?!��c���?"7
sequential/dense_1/MatMulMatMulӃ�W%��?!ro ��?0"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdamTɑ	�?!��9 $��?"!
Adam/PowPowTɑ	�?!z�r�D�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchTɑ	�?!���e��?"O
5gradient_tape/mean_squared_error/weighted_loss/Tile_1TileTɑ	�?!��ֆ�?"E
'gradient_tape/sequential/dense_1/MatMulMatMul+��6��?!��M3�?0Q      Y@Y��/Ċ�0@a�	�N]�T@q� iCTiW@yl���N�?"�
both�Your program is POTENTIALLY input-bound because 88.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�93.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 