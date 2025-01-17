�	�k�F=47@�k�F=47@!�k�F=47@	���)iؓ?���)iؓ?!���)iؓ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�k�F=47@(��9x�4@1�s����?A����K�?I�8*7Q�@Y��>rkr?*	     L@2U
Iterator::Model::ParallelMapV2jM�?!ܶm۶A@)jM�?1ܶm۶A@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate2�%䃎?!�m۶m�:@)'�����?1�m۶m3@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatg��j+��?!I�$I��4@)��~j�t�?1n۶m��0@:Preprocessing2F
Iterator::Modelݵ�|г�?!�$I�$iF@)�~j�t�x?1۶m۶m%@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice�J�4q?!      @)�J�4q?1      @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��?��?!n۶mۖK@)a��+ei?1I�$I�$@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!۶m۶m@)/n��b?1۶m۶m@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?�ܵ�|�?!������<@)a2U0*�S?1I�$I�$@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9���)iؓ?IUm�![�X@Q�|>��Y�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	(��9x�4@(��9x�4@!(��9x�4@      ��!       "	�s����?�s����?!�s����?*      ��!       2	����K�?����K�?!����K�?:	�8*7Q�@�8*7Q�@!�8*7Q�@B      ��!       J	��>rkr?��>rkr?!��>rkr?R      ��!       Z	��>rkr?��>rkr?!��>rkr?b      ��!       JGPUY���)iؓ?b qUm�![�X@y�|>��Y�?�"5
sequential/dense/MatMulMatMul�g>���?!�g>���?0"C
%gradient_tape/sequential/dense/MatMulMatMul�x7H�i�?!�j��?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul�x7H�i�?!V���P5�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�x7H�i�?!M4y���?"7
sequential/dense_1/MatMulMatMul�x7H�i�?!o��O�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�VE�@ĕ?!D���?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�VE�@ĕ?!h+���?"7
sequential/dense_2/MatMulMatMul���J�?!��t�_�?0"O
5gradient_tape/mean_squared_error/weighted_loss/Tile_1Tile�x7H�i�?!f׽!F�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�x7H�i�?!�NA�\�?Q      Y@Y>����/@aX�i��U@q����%�W@y-��j�8�?"�
both�Your program is POTENTIALLY input-bound because 88.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�95.5% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 