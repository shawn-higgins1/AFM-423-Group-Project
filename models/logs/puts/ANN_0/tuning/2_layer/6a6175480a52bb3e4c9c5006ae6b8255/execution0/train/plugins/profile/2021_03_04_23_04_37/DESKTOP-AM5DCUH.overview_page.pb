�	w,�IE{4@w,�IE{4@!w,�IE{4@	��R30ǉ?��R30ǉ?!��R30ǉ?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6w,�IE{4@�L�T�1@1�yq��?A0*��D�?I���M�@YG6ue?*	43333�H@2U
Iterator::Model::ParallelMapV2���_vO�?!̼0ɨ=@)���_vO�?1̼0ɨ=@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�!��u��?!L�q֪A<@)�~j�t��?1�Z5P8@:Preprocessing2F
Iterator::Model��_vO�?!���m��E@)_�Q�{?1� ŀ'A+@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap'�����?!+�@�`q5@)S�!�uq{?1�����*@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice����Mbp?!�<�Z5 @)����Mbp?1�<�Z5 @:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipB>�٬��?!za6�Q[L@)y�&1�l?1���^]@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor�J�4a?!�%�8k�@)�J�4a?1�%�8k�@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9��R30ǉ?I!���X@Q�Y���?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�L�T�1@�L�T�1@!�L�T�1@      ��!       "	�yq��?�yq��?!�yq��?*      ��!       2	0*��D�?0*��D�?!0*��D�?:	���M�@���M�@!���M�@B      ��!       J	G6ue?G6ue?!G6ue?R      ��!       Z	G6ue?G6ue?!G6ue?b      ��!       JGPUY��R30ǉ?b q!���X@y�Y���?�"5
sequential/dense/MatMulMatMul��d��F�?!��d��F�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�U�-%F�?!h�`tF�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul���71ͣ?!T�
~��?"E
'gradient_tape/sequential/dense_1/MatMulMatMulrbӒ̣?!0��2�	�?0"7
sequential/dense_1/MatMulMatMulrbӒ̣?!�����?0"E
)gradient_tape/sequential/dense_2/MatMul_1MatMulN�:����?!Z����?"7
sequential/dense_2/MatMulMatMulN�:����?!�V��^�?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad����R!�?!O�t��?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdamrbӒ̓?!Bp5A=��?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam*=ܲ�?!+	#���?Q      Y@Y7��Moz2@a���,daT@q)����W@yb>Ij���?"�
both�Your program is POTENTIALLY input-bound because 87.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.9 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�95.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 