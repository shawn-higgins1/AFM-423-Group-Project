�	��Ӝ�h3@��Ӝ�h3@!��Ӝ�h3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��Ӝ�h3@�X S�1@1��7�{��?A�W}�?I9{����?*	������I@2U
Iterator::Model::ParallelMapV2����Mb�?!��pl��>@)����Mb�?1��pl��>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�~j�t��?!Q�T�57@)M�O��?1��@)�3@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapF%u��?!&W�+�9@)U���N@�?1/�袋.2@:Preprocessing2F
Iterator::Model�+e�X�?!��v��F@)_�Q�{?1��h<N*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSliceŏ1w-!o?!�s��f@)ŏ1w-!o?1�s��f@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��j+���?!)�/ �K@)���_vOn?1�:��n�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!�s��f@)ŏ1w-!_?1�s��f@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIg��ap�X@Q5&����?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�X S�1@�X S�1@!�X S�1@      ��!       "	��7�{��?��7�{��?!��7�{��?*      ��!       2	�W}�?�W}�?!�W}�?:	9{����?9{����?!9{����?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qg��ap�X@y5&����?�"5
sequential/dense/MatMulMatMul��Z4`�?!��Z4`�?0"C
%gradient_tape/sequential/dense/MatMulMatMul���JǤ?!Lys���?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul���JǤ?!�~듲;�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul���JǤ?!A�R�m�?"7
sequential/dense_1/MatMulMatMul���JǤ?!6OX��?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul��x���?!�~�{��?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad���Jǔ?!�~듲;�?"E
'gradient_tape/sequential/dense_2/MatMulMatMul���Jǔ?!b�C'��?0"7
sequential/dense_2/MatMulMatMul���Jǔ?!�_D���?0"
Sum_4Sum�^�8�J�?!����E�?Q      Y@Y7��Moz2@a���,daT@qу�&�|X@y�~듲;�?"�
both�Your program is POTENTIALLY input-bound because 92.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 