�	xE𿕀7@xE𿕀7@!xE𿕀7@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-xE𿕀7@��sC�4@1+2: 	��?AyY|E�?I��K�@*	gffff�L@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate�b�=y�?!�_r ��D@)A��ǘ��?1ܤ��)_C@:Preprocessing2U
Iterator::Model::ParallelMapV2�Q���?!2���>@)�Q���?12���>@:Preprocessing2F
Iterator::Model��0�*�?!M���D@)�HP�x?1 -�^�J%@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�+e�Xw?!z�t�1�#@)���_vOn?1z�Of7�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip7�[ A�?!��hM@)�����g?1�?EI�>@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!�`3�W�@)����Mb`?1�`3�W�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapݵ�|г�?!�L���E@)a2U0*�S?1��W� @:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��H�}M?!�=��!�?)��H�}M?1�=��!�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice-C��6J?!\)t�V�?)-C��6J?1\)t�V�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI����X@Q�{�A��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	��sC�4@��sC�4@!��sC�4@      ��!       "	+2: 	��?+2: 	��?!+2: 	��?*      ��!       2	yY|E�?yY|E�?!yY|E�?:	��K�@��K�@!��K�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q����X@y�{�A��?�"5
sequential/dense/MatMulMatMuli������?!i������?0"C
%gradient_tape/sequential/dense/MatMulMatMul�:?��?!�J�Н�?0"7
sequential/dense_1/MatMulMatMul�:?��?!˶���h�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMulޱ�>�?!�F���?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��
O�Ǟ?!�e'p���?"7
sequential/dense_2/MatMulMatMul� �	��?!��eH��?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad���(�?!.�X4�?"K
$Adam/Adam/update_5/ResourceApplyAdamResourceApplyAdam�:?��?!1^��g�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�:?��?!4���ٙ�?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�:?��?!
�,M��?Q      Y@Y�C=�C=0@a��
��T@q4r�ƳW@y����iz�?"�
both�Your program is POTENTIALLY input-bound because 88.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.3 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�94.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 