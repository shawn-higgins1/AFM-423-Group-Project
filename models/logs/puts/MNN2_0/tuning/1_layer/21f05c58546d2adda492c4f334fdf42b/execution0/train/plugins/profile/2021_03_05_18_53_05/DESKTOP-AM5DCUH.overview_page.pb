�	V��#�6@V��#�6@!V��#�6@	[s���d�?[s���d�?![s���d�?"w
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails6V��#�6@@��r�"4@1�wԘs�?A�ׁsF��?I�{�?mt@Y��	j�v?*	43333sH@2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate{�G�z�?!b��,sD@)HP�sג?1�c�#\�B@:Preprocessing2U
Iterator::Model::ParallelMapV2Zd;�O��?!��,s�7@)Zd;�O��?1��,s�7@:Preprocessing2F
Iterator::Model;�O��n�?!�}��gB@)9��v��z?1�|Bٹ�*@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-C��6z?!A�9�-*@);�O��nr?1�}��g"@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip��?��?![�o�W�O@)a��+ei?1��'��[@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorŏ1w-!_?!lb��v@)ŏ1w-!_?1lb��v@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap'�����?!�#\Т�E@)Ǻ���V?1xc�	e�@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor����MbP?!��c�#\ @)����MbP?1��c�#\ @:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlicea2U0*�C?!�0�Qġ�?)a2U0*�C?1�0�Qġ�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9Ys���d�?I�~�X@Qno��Lb�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	@��r�"4@@��r�"4@!@��r�"4@      ��!       "	�wԘs�?�wԘs�?!�wԘs�?*      ��!       2	�ׁsF��?�ׁsF��?!�ׁsF��?:	�{�?mt@�{�?mt@!�{�?mt@B      ��!       J	��	j�v?��	j�v?!��	j�v?R      ��!       Z	��	j�v?��	j�v?!��	j�v?b      ��!       JGPUYYs���d�?b q�~�X@yno��Lb�?�"5
sequential/dense/MatMulMatMul���|�?!���|�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�x���?!vA0]L<�?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad��V"���?!]�ť5{�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��V"���?!��-w��?"7
sequential/dense_1/MatMulMatMul��V"���?!�x���?0"!
Adam/PowPow�x���?!���|�?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�x���?!��V"���?"E
'gradient_tape/sequential/dense_1/MatMulMatMul��e�4�?!z�� :b�?0"O
5gradient_tape/mean_squared_error/weighted_loss/Tile_1Tile�/��?!,�����?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdam 5)���?!>�| |��?Q      Y@Y{	�%��1@a�����T@q�wϷ�W@y(�[�u-�?"�
both�Your program is POTENTIALLY input-bound because 89.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�9.6 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�95.2% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 