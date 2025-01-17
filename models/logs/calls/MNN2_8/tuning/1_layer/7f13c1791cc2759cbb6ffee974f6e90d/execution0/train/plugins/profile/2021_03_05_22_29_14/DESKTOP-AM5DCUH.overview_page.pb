�	����E5@����E5@!����E5@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-����E5@�vi�a�2@1��a���?AbX9�Ȧ?I�"�dT�@*	������J@2U
Iterator::Model::ParallelMapV2ŏ1w-!�?!$I�$I�<@)ŏ1w-!�?1$I�$I�<@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate���_vO�?!}s���;@)��_vO�?1Є?�L4@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�������?!�g	7@)��ׁsF�?1�9���2@:Preprocessing2F
Iterator::Model��JY�8�?!Q^CyeD@)9��v��z?1��蛣o(@:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice����Mbp?!4�τ?@)����Mbp?14�τ?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipT㥛� �?!�����M@)�~j�t�h?1�蛣o�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��_�Le?!ȸ れ@)��_�Le?1ȸ れ@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap����Mb�?!4�τ?>@)a2U0*�S?1� れ@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�10.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI@t�=��X@Q�E;��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�vi�a�2@�vi�a�2@!�vi�a�2@      ��!       "	��a���?��a���?!��a���?*      ��!       2	bX9�Ȧ?bX9�Ȧ?!bX9�Ȧ?:	�"�dT�@�"�dT�@!�"�dT�@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q@t�=��X@y�E;��?�"5
sequential/dense/MatMulMatMul��5�zk�?!��5�zk�?0"C
%gradient_tape/sequential/dense/MatMulMatMul�����W�?!��ރa�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul��*�씝?!<�G�ƽ?"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch��t3�?!�mr� �?"7
sequential/dense_1/MatMulMatMul��t3�?!JAG��?0"!
Adam/PowPow��q��?!\��d��?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGrad��q��?!n̾��?"E
'gradient_tape/sequential/dense_1/MatMulMatMul��q��?!���1Ֆ�?0"O
5gradient_tape/mean_squared_error/weighted_loss/Tile_1Tile�{�T���?!���ܨ��?"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdamԘ��?!�c��yj�?Q      Y@Y��/Ċ�0@a�	�N]�T@q__nrX@y+��ʟ�?"�
both�Your program is POTENTIALLY input-bound because 88.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�10.4 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.8% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 