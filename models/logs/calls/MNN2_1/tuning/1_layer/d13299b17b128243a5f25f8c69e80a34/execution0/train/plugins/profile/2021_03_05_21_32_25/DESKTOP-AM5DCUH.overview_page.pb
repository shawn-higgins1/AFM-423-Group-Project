�	i�^`V(6@i�^`V(6@!i�^`V(6@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-i�^`V(6@�Z�kB�3@1�0�����?A�+e�X�?I�<L;�?*gffff�I@)       =2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate/�$��?!|��wD@)�j+��ݓ?1��"���B@:Preprocessing2U
Iterator::Model::ParallelMapV2�g��s��?!��j*��4@)�g��s��?1��j*��4@:Preprocessing2F
Iterator::Model��y�):�?!��TVSYA@)��H�}}?1�}�,@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat�J�4�?!���_0@)a��+ey?1���
�+(@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip���x�&�?!6��TVSP@)�����g?1�i,���@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor/n��b?!v�Il'@)/n��b?1v�Il'@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapA��ǘ��?!3��h.�E@)a2U0*�S?1j+����@:Preprocessing2�
NIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[1]::FromTensor��H�}M?!�}��?)��H�}M?1�}��?:Preprocessing2�
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSliceǺ���F?!Q]Eu��?)Ǻ���F?1Q]Eu��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 90.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�8.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noIS�~;�X@Q5�W ��?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�Z�kB�3@�Z�kB�3@!�Z�kB�3@      ��!       "	�0�����?�0�����?!�0�����?*      ��!       2	�+e�X�?�+e�X�?!�+e�X�?:	�<L;�?�<L;�?!�<L;�?B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb qS�~;�X@y5�W ��?�"5
sequential/dense/MatMulMatMul�0J��?!�0J��?0"C
%gradient_tape/sequential/dense/MatMulMatMul6tW�?!��'�2C�?0"Q
.gradient_tape/mean_squared_error/DynamicStitchDynamicStitch[�ra��?!>��7\��?"E
)gradient_tape/sequential/dense_1/MatMul_1MatMul�5e�?!"���?"7
sequential/dense_1/MatMulMatMul�5e�?!��Ai�?0"K
$Adam/Adam/update_3/ResourceApplyAdamResourceApplyAdamV�ى��?!p�|Z���?"!
Adam/PowPowd^��?!sJIF��?"U
4gradient_tape/sequential/dense_1/BiasAdd/BiasAddGradBiasAddGradd^��?!v�24��?"E
'gradient_tape/sequential/dense_1/MatMulMatMuld^��?!yN���?0"I
"Adam/Adam/update/ResourceApplyAdamResourceApplyAdam ��?!���N���?Q      Y@Y{	�%��1@a�����T@q5q ��hX@y!Y|�>D�?"�
both�Your program is POTENTIALLY input-bound because 90.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�8.5 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 