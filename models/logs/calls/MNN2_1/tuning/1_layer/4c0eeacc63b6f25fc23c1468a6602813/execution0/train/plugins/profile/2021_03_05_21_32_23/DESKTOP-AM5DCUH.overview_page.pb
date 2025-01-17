�	��5@��5@!��5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��5@����
3@1�^}<���?AǺ���?I�c�~�@*effff&J@)       =2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate'�����?!���yuD@)��ׁsF�?1f3%���B@:Preprocessing2U
Iterator::Model::ParallelMapV246<�R�?!�Ws_�4@)46<�R�?1�Ws_�4@:Preprocessing2F
Iterator::Modelr�����?!���`��@@)_�Q�{?1���� *@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat����Mb�?!�C"n��.@)�~j�t�x?1�����&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipe�X��?!����P@)�q����o?1ChaK��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!�C"n��@)����Mb`?1�C"n��@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�z6�>�?!QN ���E@)��_�LU?1�E����@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��H�}M?!e�㐈�?)��H�}M?1e�㐈�?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!��7j�?)Ǻ���F?1��7j�?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIa�C�i�X@Q����e�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	����
3@����
3@!����
3@      ��!       "	�^}<���?�^}<���?!�^}<���?*      ��!       2	Ǻ���?Ǻ���?!Ǻ���?:	�c�~�@�c�~�@!�c�~�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qa�C�i�X@y����e�?�"5
sequential/dense/MatMulMatMulz��:̦?!z��:̦?0"C
%gradient_tape/sequential/dense/MatMulMatMul�3���C�?!��û���?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch��@7�T�?!���+ݻ?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��@7�T�?!��+,�?"7
sequential/dense_1/MatMulMatMul��@7�T�?!�3���C�?0"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdam�3���C�?!z��:��?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad�3���C�?!��@7�T�?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdamkM�ۣe�?![5�t;�?"K
$Adam/Adam/update_1/ResourceApplyAdamResourceApplyAdamkM�ۣe�?!2���g!�?"!
Adam/PowPowkM�ۣe�?!	|���?Q      Y@Y{	�%��1@a�����T@qr�XeOmX@y|��h��?"�
both�Your program is POTENTIALLY input-bound because 87.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.7 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 