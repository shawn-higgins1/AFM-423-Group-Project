�	��7h��3@��7h��3@!��7h��3@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��7h��3@4�272@1�\T��?A��΢w*�?I��"2�b�?*	gffff�H@2U
Iterator::Model::ParallelMapV22�%䃎?!
&sp�=@)2�%䃎?1
&sp�=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatg��j+��?!A	o4u~7@)a2U0*��?1]V��F3@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap-C��6�?!%sp�9@)�&S��?1ݱ�a�E2@:Preprocessing2F
Iterator::Model�I+��?!�"NT�F@)�ZӼ�}?1�?R0��,@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice���_vOn?!#�9�@)���_vOn?1#�9�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�Pk�w�?!ݱ�a�K@)F%u�k?1���ˊ�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!�ˊ��@)�J�4a?1�ˊ��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 90.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�7.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI��w�t�X@Q��
"�"�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	4�272@4�272@!4�272@      ��!       "	�\T��?�\T��?!�\T��?*      ��!       2	��΢w*�?��΢w*�?!��΢w*�?:	��"2�b�?��"2�b�?!��"2�b�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q��w�t�X@y��
"�"�?�"5
sequential/dense/MatMulMatMulU��O=�?!U��O=�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul���먤?!�ګH�?0"C
%gradient_tape/sequential/dense/MatMulMatMul��\CF��?!�,-5�#�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��\CF��?!GlƱM�?"7
sequential/dense_1/MatMulMatMul��\CF��?!���V�w�?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul	=4�Wҙ?!P3bQ��?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam��\CF��?!��挋#�?"!
Adam/PowPow��\CF��?!\��n�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad��\CF��?!6�RU���?"7
sequential/dense_2/MatMulMatMul��\CF��?!Y���?0Q      Y@Y7��Moz2@a���,daT@qŎ��lX@y&���v#�?"�
both�Your program is POTENTIALLY input-bound because 90.6% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�7.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 