�	�덊3@�덊3@!�덊3@	䃕�Dp�?䃕�Dp�?!䃕�Dp�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6�덊3@�鷯2@1�/J�_��?A~:3P�?IF�̱���?Yw0b� ��?*	�����YI@2U
Iterator::Model::ParallelMapV2X�5�;N�?!������@@)X�5�;N�?1������@@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat9��v���?!A�A�9@)��_vO�?1��"AM5@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapg��j+��?!;�;�7@)���Q�~?1�1G���-@:Preprocessing2F
Iterator::Model�z6�>�?!�Y�j�bF@)�����w?1ؼ~�2�&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�J�4q?!z0��k� @)�J�4q?1z0��k� @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipy�&1��?! �u��K@)F%u�k?1θ	@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!4{d[@)/n��b?14{d[@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 92.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�6.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9䃕�Dp�?I	�@�}�X@QH�$X���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�鷯2@�鷯2@!�鷯2@      ��!       "	�/J�_��?�/J�_��?!�/J�_��?*      ��!       2	~:3P�?~:3P�?!~:3P�?:	F�̱���?F�̱���?!F�̱���?B      ��!       J	w0b� ��?w0b� ��?!w0b� ��?R      ��!       Z	w0b� ��?w0b� ��?!w0b� ��?b      ��!       JGPUY䃕�Dp�?b q	�@�}�X@yH�$X���?�"5
sequential/dense/MatMulMatMul;�l��P�?!;�l��P�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�v��?!a�D���?0"7
sequential/dense_1/MatMulMatMul�v��?!�h)/�0�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMul����Q�?!��$E�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��Xc�!�?!In����?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul)���-�?!���p
�?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam�v��?!\J�����?">
AssignAddVariableOp_9AssignAddVariableOp�v��?!��4�]�?"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�v��?!����g�?"E
'gradient_tape/sequential/dense_2/MatMulMatMul�v��?!f{�|��?0Q      Y@Y7��Moz2@a���,daT@qϿ����W@y�<F�|0�?"�
both�Your program is POTENTIALLY input-bound because 92.2% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�6.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 