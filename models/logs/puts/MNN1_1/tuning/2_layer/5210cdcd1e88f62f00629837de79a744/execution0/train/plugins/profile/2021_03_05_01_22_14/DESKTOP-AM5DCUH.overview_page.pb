�	og_y��8@og_y��8@!og_y��8@	��T�W��?��T�W��?!��T�W��?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6og_y��8@X�ۼ6@1����?A�������?I��]P�@Y��Tƿ�?*	������J@2U
Iterator::Model::ParallelMapV2?�ܵ�|�?!:��_�	>@)?�ܵ�|�?1:��_�	>@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat���S㥋?!����/9@)46<�R�?1��[�U4@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate��@��ǈ?!sv��6@)�St$���?1���[�.@:Preprocessing2F
Iterator::Model������?!�-�jL�E@)�ZӼ�}?1�@��~*@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSliceŏ1w-!o?!&���[@)ŏ1w-!o?1&���[@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipŏ1w-!�?!&���[L@)�����g?1�-�jL�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_�Le?!kL�*g@)��_�Le?1kL�*g@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapy�&1��?!��1�:@)ŏ1w-!_?1&���[@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��T�W��?I)�Z��X@QOٖ�zT�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	X�ۼ6@X�ۼ6@!X�ۼ6@      ��!       "	����?����?!����?*      ��!       2	�������?�������?!�������?:	��]P�@��]P�@!��]P�@B      ��!       J	��Tƿ�?��Tƿ�?!��Tƿ�?R      ��!       Z	��Tƿ�?��Tƿ�?!��Tƿ�?b      ��!       JGPUY��T�W��?b q)�Z��X@yOٖ�zT�?�"5
sequential/dense/MatMulMatMulL ���?!L ���?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul��e�G�?!]r�p�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��e�G�?!�)1���?"C
%gradient_tape/sequential/dense/MatMulMatMul����0%�?!f�ϋ�S�?0"7
sequential/dense_1/MatMulMatMul7-汐=�?!��J�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitchL ���?!���?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMulL ���?! �����?"!
Adam/PowPow��e�G�?![�ή��?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad��e�G�?!�;ۦ��?"7
sequential/dense_2/MatMulMatMul��e�G�?!h�sO��?0Q      Y@Y>����/@aX�i��U@q�f@��V@yG� w@�?"�
both�Your program is POTENTIALLY input-bound because 88.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�90.4% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 