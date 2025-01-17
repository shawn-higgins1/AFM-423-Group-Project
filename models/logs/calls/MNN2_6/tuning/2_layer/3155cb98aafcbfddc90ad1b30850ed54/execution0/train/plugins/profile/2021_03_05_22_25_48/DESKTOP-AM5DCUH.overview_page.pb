�	n��)�4@n��)�4@!n��)�4@      ��!       "n
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails-n��)�4@�n/i�.2@1K\Ǹ���?A�E���Ԩ?I}@�3iS@*	33333sJ@2U
Iterator::Model::ParallelMapV2����Mb�?!'�TA�>>@)����Mb�?1'�TA�>>@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap�q����?!%���V}=@)�+e�X�?1W��Ҍ5@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeattF��_�?!���H�~6@)M�O��?1�u=q�3@:Preprocessing2F
Iterator::Model
ףp=
�?!ۋ�<DE@)9��v��z?1���(@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice�J�4q?!(옄�@)�J�4q?1(옄�@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipŏ1w-!�?!%t�ûL@)�����g?1f]O��@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor��H�}]?!#8̺�8@)��H�}]?1#8̺�8@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 87.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).moderate"�11.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*noI�"a�'�X@Q�Z�g6�?Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�n/i�.2@�n/i�.2@!�n/i�.2@      ��!       "	K\Ǹ���?K\Ǹ���?!K\Ǹ���?*      ��!       2	�E���Ԩ?�E���Ԩ?!�E���Ԩ?:	}@�3iS@}@�3iS@!}@�3iS@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q�"a�'�X@y�Z�g6�?�"5
sequential/dense/MatMulMatMul$�Ӳ�v�?!$�Ӳ�v�?0"C
%gradient_tape/sequential/dense/MatMulMatMulY����?!>���Z7�?0"E
'gradient_tape/sequential/dense_1/MatMulMatMulY����?!j�%jI3�?0"E
)gradient_tape/sequential/dense_1/MatMul_1MatMulY����?!K�A����?"7
sequential/dense_1/MatMulMatMulY����?!�q=���?0"U
4gradient_tape/sequential/dense_2/BiasAdd/BiasAddGradBiasAddGrad�%�T���?!�k�M��?"E
)gradient_tape/sequential/dense_2/MatMul_1MatMul�%�T���?!]ЫR��?"!
Adam/PowPowY����?!�a�)�?"7
sequential/dense_2/MatMulMatMulY����?!�km�h�?0"E
'gradient_tape/sequential/dense_2/MatMulMatMul)�n.0�?!M�N����?0Q      Y@Y7��Moz2@a���,daT@q�פm{X@y�s�wf�?"�
both�Your program is POTENTIALLY input-bound because 87.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
moderate�11.0 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.no*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQb�97.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"GPU(: B 