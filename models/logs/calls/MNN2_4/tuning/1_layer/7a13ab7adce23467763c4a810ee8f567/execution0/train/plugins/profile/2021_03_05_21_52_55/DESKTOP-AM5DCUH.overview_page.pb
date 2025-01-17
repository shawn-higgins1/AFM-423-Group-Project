�	��(]P@��(]P@!��(]P@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-��(]P@�-u���N@1��D-ͭ�?A��0�*�?I��Aȗ@*	������M@2U
Iterator::Model::ParallelMapV2?�ܵ�|�?!#�u�)2;@)?�ܵ�|�?1#�u�)2;@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateHP�sג?!L�Ϻ�?@)���_vO�?1      9@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat-C��6�?!�Y7�"�5@)��_vO�?1�n0E>2@:Preprocessing2F
Iterator::Model�+e�X�?!L�ϺAC@)S�!�uq{?1)�Y7��&@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip�&S��?!�n0E�N@)����Mbp?1o0E>�@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice��H�}m?!1E>�S@)��H�}m?11E>�S@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor����Mb`?!o0E>�@)����Mb`?1o0E>�@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapn���?!�`�|֍@@)a2U0*�S?1v�)�Y7 @:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 96.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�3.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIr����X@Q��g�v�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�-u���N@�-u���N@!�-u���N@      ��!       "	��D-ͭ�?��D-ͭ�?!��D-ͭ�?*      ��!       2	��0�*�?��0�*�?!��0�*�?:	��Aȗ@��Aȗ@!��Aȗ@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qr����X@y��g�v�?�"5
sequential/dense/MatMulMatMul�e���?!�e���?0"C
%gradient_tape/sequential/dense/MatMulMatMul��Ɨ�?!xʑ�١�?0"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad`k7��0�?!P��9�?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul`k7��0�?!��n ��?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch�E�x��?!��}0��?"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdamM���'�?![`��C�?"!
Adam/PowPowM���'�?!巘g���?"7
sequential/dense_1/MatMulMatMulM���'�?!ok��L�?0"K
$Adam/Adam/update_2/ResourceApplyAdamResourceApplyAdam�K�`!��?!�x����?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdams�y�:�?!I!S�?Q      Y@Y��/Ċ�0@a�	�N]�T@qL!�%�X@y8FE4}�?"�
both�Your program is POTENTIALLY input-bound because 96.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�3.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�98.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 