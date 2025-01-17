�	x	N} �7@x	N} �7@!x	N} �7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-x	N} �7@��cw��4@1��Kǜ�?A�e��a��?II�L���@*	�����I@2U
Iterator::Model::ParallelMapV2�!��u��?!���0p<@)�!��u��?1���0p<@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat_�Qڋ?!ZEtJu;@)Zd;�O��?1������6@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateZd;�O��?!������6@)�q����?1�Ov�`/@:Preprocessing2F
Iterator::Model��_�L�?!	5�핷D@)S�!�uq{?1UD�Tw�*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice���_vOn?!zKBi{@)���_vOn?1zKBi{@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip%u��?!��[jHM@)�~j�t�h?12=���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!�*�S��@)�J�4a?1�*�S��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap-C��6�?!FA@s}9@)��_�LU?1	5�핷@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 86.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI|��\�X@Q�`���h�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��cw��4@��cw��4@!��cw��4@      ��!       "	��Kǜ�?��Kǜ�?!��Kǜ�?*      ��!       2	�e��a��?�e��a��?!�e��a��?:	I�L���@I�L���@!I�L���@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q|��\�X@y�`���h�?�"5
sequential/dense/MatMulMatMul���u��?!���u��?0"C
%gradient_tape/sequential/dense/MatMulMatMul�W۠?!8双��?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�W۠?!��8NV�?"7
sequential/dense_1/MatMulMatMul�W۠?!����a�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul۪���?!�y��?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad$��,��?!�й~G.�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul$��,��?!P�V����?"7
sequential/dense_2/MatMulMatMul$��,��?!2�r�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�ڸ���?!*F� *�?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam�Wې?!y&�U��?Q      Y@Y>����/@aX�i��U@qI���tX@y��8NV�?"�
both�Your program is POTENTIALLY input-bound because 86.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 